"""Noise that can act on long time scales: zero-frequency AR(1) branch,
spatially correlated draws, memory consistency and amplitude calibration."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
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
    config = TrainingConfig(history_steps=K, block_steps=K, batch_columns=1, slow_noise_batches=300,
                            phase_features=False)
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


def test_fast_noise_scale_multiplies_only_the_fast_branch():
    model = {"version": 1, "diffusion_gain": np.ones((K, C), np.float32)}
    runs = {}
    for scale in (0., 1., 2.):
        c = ClosureController(checkpoint(model), np.zeros(SHAPE), np.zeros(SHAPE), mode="noise",
                              correction_scale=1, tendency_clip_std=1e6, fast_noise_scale=scale, seed=4)
        runs[scale] = run(c, 1 + K)
    np.testing.assert_array_equal(runs[0.], 0)
    np.testing.assert_allclose(runs[2.], 2 * runs[1.], rtol=1e-5, atol=1e-6)
    slow = ClosureController(checkpoint(slow_model(0.5)), np.zeros(SHAPE), np.zeros(SHAPE), mode="noise",
                             correction_scale=1, tendency_clip_std=1e6, fast_noise_scale=0.)
    assert np.abs(run(slow, 1 + K)[1:]).min() > 0  # slow branch untouched by the fast scale


def test_zonal_correlation_length_recovers_generator_length():
    from neuralgcm_mz.training import zonal_correlation_length
    spherical_harmonic = pytest.importorskip("dinosaur.spherical_harmonic")
    from neuralgcm_mz.spatial import spherical_noise
    grid = spherical_harmonic.Grid.T42()
    lats = np.rad2deg(np.arcsin(grid.nodal_axes[1]))
    fields = np.asarray(spherical_noise(grid, 800)(jax.random.PRNGKey(0), 64))[..., None]
    estimate = zonal_correlation_length(fields, lats)
    assert 500 < estimate < 1300
    white = np.random.default_rng(0).normal(size=fields.shape)
    assert zonal_correlation_length(white, lats) == 0.0


def test_calibrate_picks_gate_passing_scale_closest_to_unit_spread(tmp_path, monkeypatch):
    from neuralgcm_mz import workflow
    def fake_forecast(config):
        return {}
    def fake_verify(config):
        scale = float(config["forecast_output"].split("scale_")[1].split("/")[0])
        ratio = 0.4 + 0.5 * scale  # spread grows with the fast-noise strength
        return {"leads": {d: {"spread_skill_ratio": [ratio]} for d in ("1", "3", "5")},
                "short_range_gate": {"passed": scale <= 1.0}}
    monkeypatch.setattr(workflow, "forecast", fake_forecast)
    monkeypatch.setattr(workflow, "verify", fake_verify)
    result = workflow.calibrate({"fast_noise_scales": [0.25, 0.5, 1.0, 1.5], "run_directory": str(tmp_path / "cal"),
                                 "cases": [{"issue_time": "2020-01-10"}]})
    assert result["chosen_fast_noise_scale"] == 1.0  # ratio 0.9; 1.5 fails the gate
    both = workflow.calibrate({"noise_scales": [0.25, 1.0, 1.5], "run_directory": str(tmp_path / "cal3"),
                               "cases": [{"issue_time": "2020-01-10"}]})
    assert both["chosen_noise_scale"] == 1.0 and "noise_scale" in both["candidates"][0]
    with pytest.raises(ValueError, match="exactly one"):
        workflow.calibrate({"noise_scales": [1.0], "fast_noise_scales": [1.0],
                            "run_directory": str(tmp_path / "cal4"), "cases": []})
    with pytest.raises(ValueError, match="validation"):
        workflow.calibrate({"fast_noise_scales": [1.0], "run_directory": str(tmp_path / "cal2"),
                            "cases": [{"evaluation_split": "test"}]})


def test_gain_corrects_strong_over_dispersion():
    # An undertrained sampler can be tens of times too wide; 0.2 was too tight a floor.
    rng = np.random.default_rng(1)
    mask = np.array([0, 0, 1, 1, 1, 1, 1, 1], np.float32)
    batches = [(None, rng.normal(size=(512, K, C)) * mask[None, :, None]) for _ in range(4)]
    gain = calibrate_diffusion(lambda i, h: 30 * rng.normal(size=(512, K, C)) * mask[None, :, None], batches, mask)
    np.testing.assert_allclose(gain[2:], 1 / 30, rtol=0.1)


def test_tiny_and_constant_channels_keep_their_scale():
    from neuralgcm_mz.training import _stream_channel_moments, denormalize_tendency
    rng = np.random.default_rng(2)
    array = np.stack([rng.normal(size=(20, 4, 5)), 1e-12 * rng.normal(size=(20, 4, 5)),
                      np.full((20, 4, 5), 3.0)], axis=1)
    mean, std = _stream_channel_moments(array, 0, 20, 4)
    assert 0.5 < std[0] < 1.5 and 5e-13 < std[1] < 2e-12 and std[2] < 1e-20
    # Unit-sized generated values stay at each channel's own size, never 1.
    physical = denormalize_tendency(np.ones(3, np.float32), dict(residual_mean=mean, residual_std=std))
    assert abs(physical[1] - mean[1]) < 1e-11 and abs(physical[2] - 3.0) < 1e-6


def test_untrained_sampler_stays_bounded():
    # A zero epsilon network used to grow ~1/sqrt(abar_T); x0 clipping bounds it.
    import jax.numpy as jnp
    from neuralgcm_mz.diffusion import X0_CLIP, sample
    params = jax.tree_util.tree_map(jnp.zeros_like, init_denoiser(jax.random.PRNGKey(0), H, C, K, 8))
    mask = jnp.asarray([0, 0, 1, 1, 1, 1, 1, 1], jnp.float32)
    draw = np.asarray(sample(params, jax.random.PRNGKey(1), jnp.zeros((256, H, 2 * C)), mask, steps=32))
    assert np.abs(draw).max() <= X0_CLIP + 1e-5
    np.testing.assert_array_equal(draw[:, :2], 0)


def test_solar_phase_follows_local_time():
    from neuralgcm_mz.training import solar_phase, with_phase
    times = np.array(["2020-01-01T00", "2020-01-01T06", "2020-01-01T18"], "datetime64[h]")
    p = solar_phase(times, [0., 90., 180.])
    assert p.shape == (3, 3, 2)
    np.testing.assert_allclose(p[0, 0], [0, 1], atol=1e-6)    # 00 UTC at Greenwich: local midnight
    np.testing.assert_allclose(p[1, 0], p[0, 1], atol=1e-6)   # 90E at 00 UTC == 0E at 06 UTC
    np.testing.assert_allclose(p[2, 2], p[1, 0], atol=1e-6)   # 180E at 18 UTC == 0E at 06 UTC (local 06)
    np.testing.assert_allclose(np.sum(p**2, -1), 1, atol=1e-6)
    assert with_phase(np.zeros((3, 3, 4), np.float32), times, [0., 90., 180.]).shape == (3, 3, 6)


def test_phase_checkpoint_needs_issue_time_and_uses_it():
    from neuralgcm_mz.training import PHASE_FEATURES
    value = checkpoint({"version": 1, "diffusion_gain": np.zeros((K, C), np.float32)})
    value["config"]["phase_features"] = True
    value["closure_params"] = init_closure(jax.random.PRNGKey(0), H, C, K, 8, PHASE_FEATURES)
    value["diffusion_params"] = init_denoiser(jax.random.PRNGKey(1), H, C, K, 8, PHASE_FEATURES)
    with pytest.raises(ValueError, match="issue_time"):
        ClosureController(value, np.zeros(SHAPE), np.zeros(SHAPE), mode="memory")
    means = []
    for issue in ("2020-01-01T00", "2020-01-01T12"):
        c = ClosureController(value, np.zeros(SHAPE), np.zeros(SHAPE), mode="memory", correction_scale=1,
                              tendency_clip_std=1e6, issue_time=np.datetime64(issue))
        means.append(run(c, 2)[1])
    assert np.all(np.isfinite(means)) and not np.allclose(means[0], means[1])  # memory sees the clock


def test_gaussian_skip_sampler_matches_coefficient_variance():
    from neuralgcm_mz.diffusion import sample, with_gaussian_skip
    params = init_denoiser(jax.random.PRNGKey(3), H, C, K, 8)
    params = jax.tree_util.tree_map(jnp.zeros_like, params)    # network contributes nothing
    variance = np.tile(np.linspace(0.25, 4.0, K)[:, None], (1, C)).astype(np.float32)
    params = with_gaussian_skip(params, variance, 16)
    mask = jnp.asarray(np.r_[0., np.ones(K - 1)], jnp.float32)
    history = jnp.zeros((4096, H, 2 * C))
    x = np.asarray(sample(params, jax.random.PRNGKey(4), history, mask, steps=16))
    # Few-step ancestral sampling under-disperses slightly even with the exact
    # score (THEORY: 6-13%); the per-mode gain removes that, not a 5x error.
    ratio = x[:, 1:].std(axis=0) / np.sqrt(variance[1:])
    assert np.all((ratio > 0.8) & (ratio < 1.03)), ratio
    assert np.all(x[:, 0] == 0)
