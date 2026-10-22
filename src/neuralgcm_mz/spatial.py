"""Unit-variance random fields for the closure noise, [n, latitude, longitude].

Independent column draws are spatially white: about half of their variance
lies beyond the spectral truncation and the rest is concentrated at the
smallest resolved scales, where hyperdiffusion removes it. Isotropic fields
with a finite correlation length project onto the large scales that can feed
slow variability. Every generator returns a pointwise *unit* variance, so a
column-wise DDPM or AR(1) process keeps its per-column marginal distribution;
only the cross-column covariance changes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

EARTH_RADIUS_KM = 6371.0


def independent_noise(latitudes: int, longitudes: int):
    """Spatially white reference generator (the original behaviour)."""
    def draw(key, n):
        return jax.random.normal(key, (n, latitudes, longitudes))
    return draw


def spectral_filter(total_wavenumber, length_km):
    """Gaussian-correlation amplitude per total wavenumber l.

    The variance spectrum exp(-l(l+1) L²/(2R²)) gives a correlation function
    close to exp(-d²/(2L²)) for L much smaller than the Earth radius R.
    """
    if not np.isfinite(length_km) or length_km <= 0:
        raise ValueError("Correlation length must be positive kilometres")
    l = np.asarray(total_wavenumber, dtype=np.float64)
    ratio = length_km / EARTH_RADIUS_KM
    return np.exp(-l * (l + 1) * ratio**2 / 4)


def spherical_noise(grid, length_km, *, calibration_fields=512, seed=0):
    """Isotropic fields on a Dinosaur spectral grid, returned as [n,lat,lon].

    Coefficients are white in the modal basis and filtered by total
    wavenumber. The per-latitude standard deviation is measured once from a
    fixed calibration sample, so truncation and grid details cannot change
    the requested unit variance.
    """
    _, total = grid.modal_axes
    amplitude = jnp.asarray(spectral_filter(total, length_km)[None, :] * np.asarray(grid.mask),
                            dtype=jnp.float32)
    modal_shape = tuple(grid.modal_shape)

    def raw(key, n):
        coefficients = jax.random.normal(key, (n, *modal_shape)) * amplitude
        # Dinosaur nodal arrays are [..., longitude, latitude].
        return jnp.swapaxes(grid.to_nodal(coefficients), -1, -2)

    raw = jax.jit(raw, static_argnums=1)
    sample = np.asarray(raw(jax.random.PRNGKey(seed), calibration_fields), dtype=np.float64)
    std = jnp.asarray(np.sqrt(np.mean(sample**2, axis=(0, 2)))[:, None], dtype=jnp.float32)
    if not np.all(np.asarray(std) > 0):
        raise ValueError("Correlation length removes every resolved wavenumber")

    def draw(key, n):
        return raw(key, n) / std
    return draw
