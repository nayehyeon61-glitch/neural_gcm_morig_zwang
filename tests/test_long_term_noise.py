"""Noise that can act on long time scales: zero-frequency AR(1) branch,
spatially correlated draws, memory consistency and amplitude calibration."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from neuralgcm_mz.closure import init_closure
from neuralgcm_mz.diffusion import init_denoiser
from neuralgcm_mz.rollout import ClosureController
from neuralgcm_mz.spatial import independent_noise, spectral_filter
from neuralgcm_mz.training import TrainingConfig, calibrate_diffusion, fit_slow_noise

H, K, C = 3, 8, 2
SHAPE = (H, C, 2, 3)


def checkpoint(noise_model=None, residual_mean=0.):
    value = dict(config=dict(history_steps=H, block_steps=K, dt_hours=6, max_period_hours=48, diffusion_steps=8),
                 channel_names=['t', 'u'], closure_params=init_closure(jax.random.PRNGKey(0), H, C, K, 8),
                 diffusion_params=init_denoiser(jax.random.PRNGKey(1), H, C, K, 8),
                 normalizer=dict(state_mean=np.zeros(C), state_std=np.ones(C),
                                 residual_mean=np.full(C, residual_mean), residual_std=np.ones(C)))
    if noise_model is not None:
        value["noise_model"] = noise_model
    return value


def slow_model(phi, std=1.0):
    return {"version": 1, "diffusion_gain": np.zeros((K, C), np.float32),
            "slow_cholesky": std * np.eye(C, dtype=np.float32), "slow_phi": phi}


def run(controller, steps):
    out = []
    for _ in range(steps):
        x = controller.tendency()
        controller.observe_generated(np.zeros(SHAPE[1:], np.float32), x)
        out.append(x)
    return np.stack(out)


def test_fast_only_noise_has_zero_block_integral_but_slow_branch_does_not():
    fast = ClosureController(checkpoint(), np.zeros(SHAPE), np.zeros(SHAPE), mode="noise",
                             correction_scale=1, tendency_clip_std=50)
    forcing = run(fast, 1 + 4 * K)[1:]
    assert np.abs(forcing.reshape(4, K, *SHAPE[1:]).sum(1)).max() < 1e-4 * np.abs(forcing).max()
    slow = ClosureController(checkpoint(slow_model(0.9)), np.zeros(SHAPE), np.zeros(SHAPE), mode="noise",
                             correction_scale=1, tendency_clip_std=50)
    forcing = run(slow, 1 + 4 * K)[1:]
    sums = forcing.reshape(4, K, *SHAPE[1:]).sum(1)
    assert np.abs(sums).min() > 0
    # Gain was zero, so each block is exactly constant in time.
    np.testing.assert_allclose(forcing.reshape(4, K, *SHAPE[1:]).std(1), 0, atol=1e-6)


def test_slow_branch_is_stationary_ar1_and_feeds_a_hasselmann_response():
    phi, blocks = 0.8, 600
    c = ClosureController(checkpoint(slow_model(phi)), np.zeros(SHAPE), np.zeros(SHAPE), mode="noise",
                          correction_scale=1, tendency_clip_std=50, seed=3)
    forcing = run(c, 1 + blocks * K)[1:].reshape(blocks, K, -1)[:, 0]
    assert abs(forcing.std() - 1) < 0.1
    lag = np.mean(forcing[1:] * forcing[:-1]) / np.mean(forcing**2)
    assert abs(lag - phi) < 0.08
    # dx/dt = -x/tau + f: zero-sum fast forcing integrates to (almost) nothing.
    fast = ClosureController(checkpoint(), np.zeros(SHAPE), np.zeros(SHAPE), mode="noise",
                             correction_scale=1, tendency_clip_std=50, seed=3)
    fast_forcing = run(fast, 1 + 100 * K)[1:].reshape(100 * K, -1)
    slow_forcing = run(c, 100 * K).reshape(100 * K, -1)

    def response(f, tau=40.):
        x = np.zeros(f.shape[1]); trace = []
        for value in f:
            x = x + (-x / tau + value); trace.append(x)
        return np.stack(trace)[200:].std() / f.std()
    assert response(slow_forcing) > 5 * response(fast_forcing)


def test_memory_stores_unscaled_estimate_and_neutral_first_interval():
    c = ClosureController(checkpoint(slow_model(0.5), residual_mean=2.), np.zeros(SHAPE), np.zeros(SHAPE),
                          mode="full", correction_scale=0.1)
    first = c.tendency()
    c.observe_generated(np.zeros(SHAPE[1:], np.float32), first)
    np.testing.assert_array_equal(c.corrections[-1], 2.)  # training mean, normalizes to 0
    applied = c.tendency()
    c.observe_generated(np.zeros(SHAPE[1:], np.float32), applied)
    np.testing.assert_allclose(c.corrections[-1], applied / 0.1, rtol=1e-5)


def test_correlated_noise_fn_correlates_columns_and_gain_scales_fast_noise():
    shared = lambda key, n: np.broadcast_to(np.asarray(jax.random.normal(key, (n, 1, 1))), (n, 2, 3))
    model = {"version": 1, "diffusion_gain": np.ones((K, C), np.float32)}
    c = ClosureController(checkpoint(model), np.zeros(SHAPE), np.zeros(SHAPE), mode="noise",
                          correction_scale=1, tendency_clip_std=50, noise_fn=shared)
    forcing = run(c, 1 + K)[1:]
    assert np.abs(forcing).max() > 0
    np.testing.assert_allclose(forcing, forcing[..., :1, :1] * np.ones_like(forcing), atol=1e-5)
    off = ClosureController(checkpoint({"version": 1, "diffusion_gain": np.zeros((K, C), np.float32)}),
                            np.zeros(SHAPE), np.zeros(SHAPE), mode="noise", correction_scale=1)
    np.testing.assert_array_equal(run(off, 1 + K), 0)


def test_mode_noise_excludes_closure_mean():
    memory = ClosureController(checkpoint(slow_model(0.)), np.zeros(SHAPE), np.ones(SHAPE), mode="memory",
                               correction_scale=1)
    assert np.abs(run(memory, 3)).max() > 0
    gain_zero = {"version": 1, "diffusion_gain": np.zeros((K, C), np.float32)}
    noise = ClosureController(checkpoint(gain_zero), np.zeros(SHAPE), np.ones(SHAPE), mode="noise",
                              correction_scale=1)
    np.testing.assert_array_equal(run(noise, 3), 0)


def test_calibration_gain_and_slow_fit_recover_known_statistics():
    rng = np.random.default_rng(0)
    mask = np.array([0, 0, 1, 1, 1, 1, 1, 1], np.float32)
    batches = [(None, rng.normal(size=(512, K, C)) * mask[None, :, None]) for _ in range(4)]
    gain = calibrate_diffusion(lambda i, h: 2 * rng.normal(size=(512, K, C)) * mask[None, :, None], batches, mask)
    np.testing.assert_allclose(gain[2:], 0.5, atol=0.05)
    np.testing.assert_array_equal(gain[:2], 1)
    # Synthetic archive whose residual block means follow AR(1) with phi=0.6.
    n, phi = 4000, 0.6
    means = np.zeros(n // K)
    for i in range(1, len(means)):
        means[i] = phi * means[i-1] + np.sqrt(1 - phi**2) * rng.normal()
    residual = np.repeat(means, K)[:, None, None, None]
    archive = SimpleNamespace(states=np.zeros((n, 1, 1, 1)), residual=residual, latitudes=np.array([0.]))
    norm = {"state_mean": np.zeros(1), "state_std": np.ones(1), "residual_mean": np.zeros(1), "residual_std": np.ones(1)}
    config = TrainingConfig(history_steps=K, block_steps=K, batch_columns=1, slow_noise_batches=300)
    origins = range(K, n - 2 * K + 1, K)  # block-aligned so each target block is one AR(1) value
    fit = fit_slow_noise(archive, origins, config, norm, rng, lambda h, t: t)
    assert abs(fit["slow_phi"] - phi) < 0.12
    assert abs(float(fit["slow_cholesky"][0, 0]) - 1) < 0.15


def test_spectral_filter_and_spherical_noise_unit_variance():
    l = np.arange(64)
    f = spectral_filter(l, 500)
    assert f[0] == 1 and np.all(np.diff(f) < 0)
    with pytest.raises(ValueError):
        spectral_filter(l, 0)
    spherical_harmonic = pytest.importorskip("dinosaur.spherical_harmonic")
    from neuralgcm_mz.spatial import spherical_noise
    grid = spherical_harmonic.Grid.T21()
    fields = np.asarray(spherical_noise(grid, 1000)(jax.random.PRNGKey(1), 400))
    lat, lon = grid.nodal_shape[1], grid.nodal_shape[0]
    assert fields.shape == (400, lat, lon)
    np.testing.assert_allclose(fields.var(axis=(0, 2)), 1, rtol=0.2)
    neighbour = np.mean(fields[:, :, 1:] * fields[:, :, :-1])
    white = np.asarray(independent_noise(lat, lon)(jax.random.PRNGKey(1), 400))
    assert neighbour > 0.5 and abs(np.mean(white[:, :, 1:] * white[:, :, :-1])) < 0.05
