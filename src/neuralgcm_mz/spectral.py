"""Orthonormal DCT-II on physical time, never on channel or spatial axes."""
import numpy as np
import jax.numpy as jnp


def dct_matrix(n, dtype=jnp.float32):
    if n < 2:
        raise ValueError("DCT requires at least two actual time samples")
    k = jnp.arange(n)[:, None]
    t = jnp.arange(n)[None, :]
    matrix = jnp.cos(jnp.pi * (t + 0.5) * k / n) * jnp.sqrt(2.0 / n)
    return matrix.at[0].multiply(1 / jnp.sqrt(2.0)).astype(dtype)


def dct(x, axis=1):
    x = jnp.moveaxis(x, axis, -1)
    out = x @ dct_matrix(x.shape[-1], x.dtype).T
    return jnp.moveaxis(out, -1, axis)


def idct(x, axis=1):
    x = jnp.moveaxis(x, axis, -1)
    out = x @ dct_matrix(x.shape[-1], x.dtype)
    return jnp.moveaxis(out, -1, axis)


def highpass_mask(n, dt_hours, max_period_hours):
    """DCT k>0 represents period 2*n*dt/k. DC is always excluded.

    A strict high pass is NOT white weather forcing: its block integral is
    zero. Low-frequency correction is retained by the learned history mean.
    """
    if n < 2 or dt_hours <= 0 or max_period_hours <= 2 * dt_hours:
        raise ValueError("Need n>=2 and cutoff period strictly above Nyquist period")
    k = np.arange(n)
    mask = (k > 0) & (k * max_period_hours >= 2 * n * dt_hours)
    if not mask.any():
        raise ValueError("No DCT mode passes the requested temporal cutoff")
    return jnp.asarray(mask, dtype=jnp.float32)


def split_bands(x, dt_hours, max_period_hours, axis=1):
    coeff = dct(x, axis)
    shape = [1] * coeff.ndim
    shape[axis] = coeff.shape[axis]
    mask = highpass_mask(coeff.shape[axis], dt_hours, max_period_hours).reshape(shape)
    fast = idct(coeff * mask, axis)
    return x - fast, fast
