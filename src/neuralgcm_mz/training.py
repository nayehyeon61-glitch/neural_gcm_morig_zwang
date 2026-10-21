"""Causal, split-safe offline fitting of a history closure and residual DDPM.

The baseline remains frozen. This is residual-tendency fitting, not a claim of
end-to-end NeuralGCM training or improved 40-day skill. A separate validation
rollout must select correction strength before using a held-out test period.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    history_steps: int = 8
    block_steps: int = 8
    dt_hours: float = 6.0
    max_period_hours: float = 48.0
    hidden: int = 128
    diffusion_steps: int = 32
    batch_columns: int = 64
    closure_updates: int = 1000
    diffusion_updates: int = 2000
    validation_every: int = 50
    validation_batches: int = 4
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    seed: int = 42
    memory_limit_gb: float = 96.0
    normalization_chunk_steps: int = 4
    calibration_batches: int = 16
    slow_noise_batches: int = 64
    correlation_origins: int = 8

    def __post_init__(self):
        positive_int = (
            "history_steps", "block_steps", "hidden", "diffusion_steps",
            "batch_columns", "closure_updates", "diffusion_updates",
            "validation_every", "validation_batches", "normalization_chunk_steps",
            "calibration_batches", "slow_noise_batches", "correlation_origins",
        )
        for name in positive_int:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.diffusion_steps < 2 or self.block_steps < 2:
            raise ValueError("diffusion_steps and block_steps must be at least 2")
        for name in ("dt_hours", "max_period_hours", "learning_rate", "gradient_clip", "memory_limit_gb"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.memory_limit_gb > 96:
            raise ValueError("memory_limit_gb cannot exceed the requested 96 GB cap")
        # DCT-II bins k/(2*N*dt); k=N-1 is the highest representable bin.
        shortest = 2 * self.block_steps * self.dt_hours / (self.block_steps - 1)
        if self.max_period_hours < shortest:
            raise ValueError(f"No DCT bin is retained; max_period_hours must be >= {shortest:g}")

    @classmethod
    def from_json(cls, path: str | Path):
        values = json.loads(Path(path).read_text())
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a nested array/JSON tree without executable pickle data."""
    arrays: dict[str, np.ndarray] = {}

    def encode(value):
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("Checkpoint dictionary keys must be strings")
            return {"type": "dict", "items": [[k, encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"type": "tuple" if isinstance(value, tuple) else "list", "items": [encode(v) for v in value]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return {"type": "json", "value": value}
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays are forbidden in checkpoints")
        key = f"array_{len(arrays):05d}"
        arrays[key] = array
        return {"type": "array", "key": key}

    spec = encode(payload)
    arrays["__tree__"] = np.asarray(json.dumps({"format_version": 1, "tree": spec}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temp = Path(stream.name)
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read only arrays and a restricted JSON container description."""
    with np.load(path, allow_pickle=False) as archive:
        schema = json.loads(str(archive["__tree__"]))
        if schema.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")

        def decode(spec):
            kind = spec["type"]
            if kind == "dict":
                return {k: decode(v) for k, v in spec["items"]}
            if kind in ("tuple", "list"):
                values = [decode(v) for v in spec["items"]]
                return tuple(values) if kind == "tuple" else values
            if kind == "json":
                return spec["value"]
            if kind == "array":
                return archive[spec["key"]].copy()
            raise ValueError(f"Unknown checkpoint node type: {kind}")

        return decode(schema["tree"])


def normalize_history(states, past_residuals, normalizer):
    """Normalize [..., H, C] histories with train-only channel statistics."""
    states = np.asarray(states, dtype=np.float32)
    past_residuals = np.asarray(past_residuals, dtype=np.float32)
    if states.shape != past_residuals.shape:
        raise ValueError("State and previous-residual histories must have matching shapes")
    state = (states - normalizer["state_mean"]) / normalizer["state_std"]
    residual = (past_residuals - normalizer["residual_mean"]) / normalizer["residual_std"]
    return np.concatenate((state, residual), axis=-1).astype(np.float32)


def denormalize_tendency(values, normalizer):
    return values * normalizer["residual_std"] + normalizer["residual_mean"]


def _stream_channel_moments(array, start: int, stop: int, chunk_steps: int):
    """Parallel/Welford moments, holding only a few time slices in memory."""
    count = 0
    mean = np.zeros(array.shape[1], np.float64)
    m2 = np.zeros_like(mean)
    for left in range(start, stop, chunk_steps):
        block = np.asarray(array[left:min(stop, left + chunk_steps)], dtype=np.float64)
        if not np.all(np.isfinite(block)):
            raise ValueError("Training data contain NaN or infinity")
        values = block.transpose(1, 0, 2, 3).reshape(array.shape[1], -1)
        n = values.shape[1]
        chunk_mean = values.mean(axis=1)
        chunk_m2 = ((values - chunk_mean[:, None]) ** 2).sum(axis=1)
        delta = chunk_mean - mean
        total = count + n
        m2 += chunk_m2 + delta**2 * count * n / total
        mean += delta * n / total
        count = total
    if count == 0:
        raise ValueError("Training split is empty")
    std = np.sqrt(m2 / count)
    # Constant channels remain finite without amplifying small numerical errors.
    std = np.where(std > 1e-12, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _check_memory(config: TrainingConfig, shape):
    """Conservative transient estimate; process RSS checked during updates too."""
    import psutil
    _, channels, nlat, nlon = shape
    normalization_bytes = config.normalization_chunk_steps * channels * nlat * nlon * 8 * 5
    batch_bytes = config.batch_columns * (2 * config.history_steps + config.block_steps) * channels * 4
    # Parameters, gradients, Adam buffers, JAX execution and compiler allowance.
    model_bytes = (2 * config.history_steps + 2 * config.block_steps) * channels * config.hidden * 4 * 16
    estimate = normalization_bytes + batch_bytes * 32 + model_bytes + 512 * 1024**2
    limit = config.memory_limit_gb * 1e9
    if estimate > 0.8 * limit:
        raise MemoryError(f"Estimated working set {estimate / 1e9:.2f} GB exceeds 80% of {config.memory_limit_gb:g} GB; reduce normalization_chunk_steps/batch_columns/hidden")
    if estimate > psutil.virtual_memory().available:
        raise MemoryError("Insufficient available host memory for estimated working set")
    return {"estimated_working_set_gb": estimate / 1e9, "limit_gb": config.memory_limit_gb}


def _check_rss(limit_gb):
    import psutil
    try:
        rss = psutil.Process().memory_info().rss
    except (psutil.Error, OSError):
        # Some container PID namespaces expose a host /proc. getrusage reads
        # this process directly; Linux reports its peak resident set in KiB.
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
    if rss > 0.9 * limit_gb * 1e9:
        raise MemoryError(f"Process RSS {rss/1e9:.2f} GB reached the safety margin for {limit_gb:g} GB")


def _draw_columns(archive, origins, config, rng):
    if len(origins) == 0:
        raise ValueError("Split is too short for history and future block")
    origin = origins[int(rng.integers(len(origins)))]
    ny, nx = archive.states.shape[-2:]
    weights = np.maximum(np.cos(np.deg2rad(archive.latitudes)), 0)
    if weights.sum() <= 0:
        raise ValueError("Latitude grid has no positive area")
    ys = rng.choice(ny, config.batch_columns, p=weights / weights.sum())
    xs = rng.integers(nx, size=config.batch_columns)
    return origin, ys, xs


def sample_batch(archive, origins, config, normalizer, rng):
    """Read random columns from one origin; never materialize a global window.

    Each past residual ends at its corresponding conditioning state. In
    particular residual[origin] is a target and cannot appear in the history.
    Latitude sampling is proportional to cos(latitude), for area-weighted
    fitting on latitude-longitude grids.
    """
    return _window(archive, *_draw_columns(archive, origins, config, rng), config, normalizer)


def _window(archive, origin, ys, xs, config, normalizer):
    h, k = config.history_steps, config.block_steps
    # Slice time first: NumPy mixed advanced indexing would otherwise move the
    # column axis to the front depending on the exact expression.
    states = np.asarray(archive.states[origin-h+1:origin+1][:, :, ys, xs]).transpose(2, 0, 1)
    past = np.asarray(archive.residual[origin-h:origin][:, :, ys, xs]).transpose(2, 0, 1)
    future = np.asarray(archive.residual[origin:origin+k][:, :, ys, xs]).transpose(2, 0, 1)
    history = normalize_history(states, past, normalizer)
    target = ((future - normalizer["residual_mean"]) / normalizer["residual_std"]).astype(np.float32)
    if not np.isfinite(history).all() or not np.isfinite(target).all():
        raise ValueError("Selected training/validation window contains non-finite values")
    return history, target


def fit_slow_noise(archive, origins, config, normalizer, rng, innovation):
    """Covariance and block-to-block memory of the block-mean innovation.

    The fast DDPM sees only DCT modes k>0, whose sum over each block is
    exactly zero. The remaining block mean of ``target - closure mean`` is
    the zero-frequency forcing that a Hasselmann-type slow response
    integrates. It is modelled as a vector AR(1) process over consecutive
    blocks, s' = phi s + sqrt(1-phi^2) L z, with L the Cholesky factor of the
    channel covariance and one pooled phi (keeps the process stationary).
    ``origins`` must allow two consecutive blocks inside the split.
    """
    k = config.block_steps
    first, second = [], []
    for _ in range(config.slow_noise_batches):
        origin, ys, xs = _draw_columns(archive, origins, config, rng)
        for target_list, start in ((first, origin), (second, origin + k)):
            history, target = _window(archive, start, ys, xs, config, normalizer)
            target_list.append(np.asarray(innovation(history, target)).mean(axis=1))
    a = np.concatenate(first).astype(np.float64)
    b = np.concatenate(second).astype(np.float64)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    covariance = a.T @ a / len(a)
    scale = max(float(np.mean(np.diag(covariance))), 1e-12)
    cholesky = np.linalg.cholesky(covariance + 1e-6 * scale * np.eye(len(covariance)))
    phi = float(np.clip(np.sum(a * b) / max(np.sum(a * a), 1e-12), 0.0, 0.99))
    return {"slow_cholesky": cholesky.astype(np.float32), "slow_phi": phi,
            "slow_std": np.sqrt(np.diag(covariance)).astype(np.float32),
            "slow_samples": int(len(a))}


def zonal_correlation_length(fields, latitudes, max_latitude=60.):
    """Gaussian e-folding length (km) of zonal correlation, [sample,lat,lon,channel].

    Each sample and channel is standardized over the globe. For every zonal
    lag with correlation c in (0.05, 0.95), L = d / sqrt(-2 ln c) at that
    lag's great-circle-chord distance d; the median over latitudes within
    ``max_latitude`` and lags is returned. 0 means grid-scale white noise.
    """
    from .spatial import EARTH_RADIUS_KM
    f = np.asarray(fields, np.float64)
    f = (f - f.mean(axis=(1, 2), keepdims=True)) / np.maximum(f.std(axis=(1, 2), keepdims=True), 1e-12)
    nx = f.shape[2]
    lengths = []
    for j, lat in enumerate(np.asarray(latitudes, np.float64)):
        if abs(lat) > max_latitude:
            continue
        row = f[:, j]
        for lag in range(1, max(2, nx // 4)):
            c = float(np.mean(row * np.roll(row, lag, axis=1)))
            if not 0.05 < c < 0.95:
                if c <= 0.05:
                    break
                continue
            d = EARTH_RADIUS_KM * np.cos(np.deg2rad(lat)) * 2 * np.pi * lag / nx
            lengths.append(d / np.sqrt(-2 * np.log(c)))
    return float(np.median(lengths)) if lengths else 0.0


def fit_spatial_correlation(archive, origins, config, normalizer, rng, innovation, mask):
    """Correlation lengths of the fast and block-mean innovation, from data.

    The DDPM and AR(1) branches are column-wise, so their spatial structure
    comes only from the random fields that drive them. Fitting the length
    to training innovations replaces an arbitrary choice and sets how much
    of the calibrated per-column variance reaches resolved scales.
    """
    from .spectral import dct, idct
    ny, nx = archive.states.shape[-2:]
    ys, xs = np.repeat(np.arange(ny), nx), np.tile(np.arange(nx), ny)
    fast, slow = [], []
    for _ in range(config.correlation_origins):
        origin = origins[int(rng.integers(len(origins)))]
        history, target = _window(archive, origin, ys, xs, config, normalizer)
        e = np.asarray(innovation(history, target))
        f = np.asarray(idct(dct(e, axis=1) * np.asarray(mask)[None, :, None], axis=1))
        fast.append(f.reshape(ny, nx, *f.shape[1:]).transpose(2, 0, 1, 3))
        slow.append(e.mean(axis=1).reshape(1, ny, nx, -1))
    return {"correlation_km_fast": zonal_correlation_length(np.concatenate(fast), archive.latitudes),
            "correlation_km_slow": zonal_correlation_length(np.concatenate(slow), archive.latitudes)}


def calibrate_diffusion(sampler, batches, mask):
    """Per (mode, channel) gain matching DDPM sample spread to held-out innovations.

    A small denoiser with few reverse steps does not reproduce the target
    variance (both inflation and sampler under-dispersion occur). The gain
    is fitted on validation windows only and clipped to [0.2, 5].
    """
    target_power = 0.
    sample_power = 0.
    for i, (history, coefficients) in enumerate(batches):
        draw = np.asarray(sampler(i, history))
        target_power = target_power + np.sum(np.asarray(coefficients, np.float64)**2, axis=0)
        sample_power = sample_power + np.sum(draw.astype(np.float64)**2, axis=0)
    active = np.asarray(mask)[:, None] > 0
    ratio = np.sqrt(target_power / np.maximum(sample_power, 1e-30))
    gain = np.where(active, np.clip(ratio, 0.2, 5.0), 1.0)
    return gain.astype(np.float32)


def train(archive_path, output_path, config: TrainingConfig | None = None):
    """Fit both stages, select by held-out validation loss, write safe checkpoint.

    Training/validation targets are allowed only during fitting. The test split
    is never opened for normalization or model selection. The baseline is frozen
    and closure outputs are residual tendencies in per-second state units.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from .closure import init_closure, predict_closure
    from .data import ResidualArchive
    from .diffusion import init_denoiser, diffusion_loss
    from .spectral import dct, highpass_mask

    config = config or TrainingConfig()
    archive = ResidualArchive(archive_path)
    if not np.isclose(archive.dt_seconds, config.dt_hours * 3600, rtol=0, atol=1e-6):
        raise ValueError("Training cadence differs from actual archive timestamps")
    train_origins = archive.window_origins("train", config.history_steps, config.block_steps)
    validation_origins = archive.window_origins("validation", config.history_steps, config.block_steps)
    if not train_origins or not validation_origins:
        raise ValueError("Train and validation splits each need history_steps + block_steps samples")
    memory = _check_memory(config, archive.states.shape)
    start, stop = archive.splits["train"]
    normalizer = {}
    for source, prefix in ((archive.states, "state"), (archive.residual, "residual")):
        mean, std = _stream_channel_moments(source, start, stop, config.normalization_chunk_steps)
        normalizer[f"{prefix}_mean"] = mean
        normalizer[f"{prefix}_std"] = std
    channels = archive.states.shape[1]
    key = jax.random.PRNGKey(config.seed)
    key, closure_key, diffusion_key = jax.random.split(key, 3)
    closure_params = init_closure(closure_key, config.history_steps, channels, config.block_steps, config.hidden)
    diffusion_params = init_denoiser(diffusion_key, config.history_steps, channels, config.block_steps, config.hidden)
    mask = highpass_mask(config.block_steps, config.dt_hours, config.max_period_hours)
    if not np.asarray(mask).any():
        raise ValueError("No high-frequency DCT modes selected")
    optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip), optax.adam(config.learning_rate))

    def closure_loss(params, history, target):
        return jnp.mean((predict_closure(params, history) - target)**2)

    @jax.jit
    def closure_update(params, optimizer_state, history, target):
        loss, grad = jax.value_and_grad(closure_loss)(params, history, target)
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
        return optax.apply_updates(params, updates), optimizer_state, loss

    closure_eval = jax.jit(closure_loss)
    rng = np.random.default_rng(config.seed)
    # Fixed validation examples and diffusion random keys allow comparable
    # model selection; they are never used for an optimizer update.
    vrng = np.random.default_rng(config.seed + 1)
    validation = [sample_batch(archive, validation_origins, config, normalizer, vrng)
                  for _ in range(config.validation_batches)]
    trace = []
    state = optimizer.init(closure_params)
    best_value = float("inf")
    best_closure = closure_params
    for step in range(1, config.closure_updates + 1):
        history, target = sample_batch(archive, train_origins, config, normalizer, rng)
        closure_params, state, loss = closure_update(closure_params, state, history, target)
        if step % config.validation_every == 0 or step == config.closure_updates:
            train_loss = float(loss)
            value = float(np.mean([float(closure_eval(closure_params, h, t)) for h, t in validation]))
            if not np.isfinite(train_loss) or not np.isfinite(value):
                raise FloatingPointError("Non-finite closure training or validation loss")
            if value < best_value:
                best_value, best_closure = value, closure_params
            trace.append({"stage": "closure", "update": step, "train_loss": train_loss, "validation_loss": value})
            _check_rss(config.memory_limit_gb)
    closure_params = best_closure
    closure_validation_loss = best_value
    del state

    @jax.jit
    def innovation_coefficients(history, target):
        # The deterministic forecast uses history only; future residuals enter
        # solely as diffusion targets, never as conditioning information.
        innovation = target - predict_closure(closure_params, history)
        return dct(innovation, axis=1) * jnp.asarray(mask)[None, :, None]

    def ddpm_loss(params, noise_key, coefficients, history):
        return diffusion_loss(params, noise_key, coefficients, history, mask, steps=config.diffusion_steps)

    @jax.jit
    def diffusion_update(params, optimizer_state, noise_key, coefficients, history):
        loss, grad = jax.value_and_grad(ddpm_loss)(params, noise_key, coefficients, history)
        updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
        return optax.apply_updates(params, updates), optimizer_state, loss

    diffusion_eval = jax.jit(ddpm_loss)
    validation_coeff = [(h, innovation_coefficients(h, t)) for h, t in validation]
    validation_keys = jax.random.split(jax.random.PRNGKey(config.seed + 2), len(validation))
    state = optimizer.init(diffusion_params)
    best_value = float("inf")
    best_diffusion = diffusion_params
    for step in range(1, config.diffusion_updates + 1):
        history, target = sample_batch(archive, train_origins, config, normalizer, rng)
        key, noise_key = jax.random.split(key)
        coefficients = innovation_coefficients(history, target)
        diffusion_params, state, loss = diffusion_update(diffusion_params, state, noise_key, coefficients, history)
        if step % config.validation_every == 0 or step == config.diffusion_updates:
            train_loss = float(loss)
            value = float(np.mean([float(diffusion_eval(diffusion_params, vk, coeff, h))
                                   for vk, (h, coeff) in zip(validation_keys, validation_coeff)]))
            if not np.isfinite(train_loss) or not np.isfinite(value):
                raise FloatingPointError("Non-finite diffusion training or validation loss")
            if value < best_value:
                best_value, best_diffusion = value, diffusion_params
            trace.append({"stage": "diffusion", "update": step, "train_loss": train_loss, "validation_loss": value})
            _check_rss(config.memory_limit_gb)
    # Noise calibration: amplitude per mode and the zero-frequency AR(1) branch.
    from .diffusion import sample as ddpm_sample
    crng = np.random.default_rng(config.seed + 3)
    calibration = [sample_batch(archive, validation_origins, config, normalizer, crng)
                   for _ in range(config.calibration_batches)]
    draw = jax.jit(lambda k, h: ddpm_sample(best_diffusion, k, h, mask, config.diffusion_steps))
    calibration_key = jax.random.PRNGKey(config.seed + 4)
    diffusion_gain = calibrate_diffusion(
        lambda i, h: draw(jax.random.fold_in(calibration_key, i), h),
        [(h, innovation_coefficients(h, t)) for h, t in calibration], mask)
    pair_origins = archive.window_origins("train", config.history_steps, 2 * config.block_steps)
    if not pair_origins:
        raise ValueError("Training split needs history_steps + 2*block_steps samples for slow noise")
    innovation = jax.jit(lambda h, t: t - predict_closure(closure_params, h))
    slow = fit_slow_noise(archive, pair_origins, config, normalizer,
                          np.random.default_rng(config.seed + 5), innovation)
    spatial = fit_spatial_correlation(archive, train_origins, config, normalizer,
                                      np.random.default_rng(config.seed + 6), innovation, mask)
    noise_model = {"version": 1, "diffusion_gain": diffusion_gain, **slow, **spatial}
    summary = {
        "closure_validation_loss": closure_validation_loss,
        "noise_calibration": {"diffusion_gain_min": float(np.min(diffusion_gain)),
                              "diffusion_gain_max": float(np.max(diffusion_gain)),
                              "slow_phi": slow["slow_phi"],
                              "slow_std_mean": float(np.mean(slow["slow_std"])),
                              "slow_samples": slow["slow_samples"], **spatial,
                              "gain_fit_split": "validation", "slow_fit_split": "train"},
        "diffusion_validation_loss": best_value,
        "normalization_fit_split": "train", "train_bounds": [start, stop],
        "train_windows": len(train_origins), "validation_windows": len(validation_origins),
        "validation_selection": "fixed sampled columns and fixed diffusion noise keys",
        "test_used": False,
        "memory": memory,
        "trace": trace,
        "limitation": "Offline one-step residual training; closed-loop 40-day skill and short-range non-inferiority require separate validation/test rollouts.",
    }
    checkpoint = {
        "config": asdict(config),
        "closure_params": closure_params,
        "diffusion_params": best_diffusion,
        "noise_model": noise_model,
        "normalizer": normalizer,
        "channel_names": list(archive.channel_names),
        "state_units": archive.manifest["state_units"],
        "residual_units": archive.manifest["residual_units"],
        "dt_seconds": archive.dt_seconds,
        "provenance": {**archive.manifest.get("provenance", {}),
            "split_time_bounds": {name: [str(archive.timestamps[a]),
                str(archive.timestamps[b-1]+np.timedelta64(round(archive.dt_seconds*1e9),'ns'))]
                for name,(a,b) in archive.splits.items()}},
        "training_summary": summary,
    }
    save_checkpoint(output_path, checkpoint)
    report_path = Path(output_path).with_suffix(".training.json")
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
