"""Long-lead response of the real NeuralGCM TL63 core to the closure noise.

Software/mechanism experiment, not a skill evaluation: the denoiser and
closure are untrained and the forcing amplitude is prescribed. All runs start
from the same encoded state and share NeuralGCM's internal random stream, so
differences from the unforced baseline are caused by the injected forcing.

Configurations (temperature-only forcing, mode="noise", equal nominal scale):
  fast_white    previous design: zero-sum DCT DDPM, independent columns
  fast_corr     same DDPM with 500 km correlated draws
  slow_corr     zero-frequency AR(1) branch only, 500 km correlated
  full_corr     fast_corr + slow_corr (the new default)
  ic_tiny       no forcing, 0.01 K initial perturbation (intrinsic chaos)
  full_corr_sX  full_corr with fast_noise_scale X (--fast-scales)

Usage: python experiments/noise_response_tl63.py --days 20 --output runs/noise_response.json
       python experiments/noise_response_tl63.py --days 5 --only fast_white --fast-scales 0.25 0.5 0.75 --output ...
"""
import argparse
import json
import time

import jax
import numpy as np

from neuralgcm_mz.closure import init_closure
from neuralgcm_mz.diffusion import init_denoiser
from neuralgcm_mz.rollout import ClosureController

H, K = 4, 8
DT = 6 * 3600.
RESIDUAL_STD = 1e-5  # K/s, about 0.86 K/day


def make_checkpoint(channels, *, gain, slow):
    closure = jax.tree_util.tree_map(np.zeros_like, init_closure(jax.random.PRNGKey(0), H, channels, K, 32))
    model = {"version": 1, "diffusion_gain": np.full((K, channels), gain, np.float32)}
    if slow:
        model.update(slow_cholesky=np.eye(channels, dtype=np.float32), slow_phi=0.5)
    return {"config": dict(history_steps=H, block_steps=K, dt_hours=6, max_period_hours=48, diffusion_steps=16),
            "channel_names": [f"temperature:sigma_{i}" for i in range(channels)],
            "closure_params": closure,
            "diffusion_params": init_denoiser(jax.random.PRNGKey(1), H, channels, K, 32),
            "noise_model": model,
            "normalizer": dict(state_mean=np.zeros(channels), state_std=np.full(channels, 50.),
                               residual_mean=np.zeros(channels), residual_std=np.full(channels, RESIDUAL_STD))}


def main():
    import neuralgcm
    from neuralgcm import demo
    from neuralgcm_mz.neuralgcm_backend import NeuralGCMBackend
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", nargs="*", help="run only these configurations (ic_tiny included by name)")
    parser.add_argument("--fast-scales", nargs="*", type=float, default=[])
    args = parser.parse_args()
    model = neuralgcm.PressureLevelModel.from_checkpoint(demo.load_checkpoint_tl63_stochastic())
    backend = NeuralGCMBackend(model)
    data = demo.load_data(model.data_coords).isel(time=0)
    forcing = model.forcings_from_xarray(data)
    initial = backend.encode(model.inputs_from_xarray(data), forcing, rng_key=jax.random.PRNGKey(0))
    sigma, lon, lat = backend.sigma_shape
    weights = np.cos(np.asarray(model.model_coords.horizontal.latitudes))[:, None] * np.ones((1, lon))
    weights = weights / weights.sum()
    correlated = backend.spherical_noise(500)
    temperature = lambda s: np.asarray(backend.sigma_fields(s)["temperature"]).transpose(0, 2, 1)  # [sigma,lat,lon]
    history = np.repeat(temperature(initial)[None], H, axis=0)
    configs = {
        "fast_white": dict(gain=1., slow=False, noise_fn=None),
        "fast_corr": dict(gain=1., slow=False, noise_fn=correlated),
        "slow_corr": dict(gain=0., slow=True, noise_fn=correlated),
        "full_corr": dict(gain=1., slow=True, noise_fn=correlated),
    }
    if args.only is not None:
        configs = {name: spec for name, spec in configs.items() if name in args.only}
    for scale in args.fast_scales:
        configs[f"full_corr_s{scale:g}"] = dict(gain=1., slow=True, noise_fn=correlated, fast_scale=scale)
    steps = args.days * 4
    report_days = [d for d in (1, 3, 5, 10, 15, 20, 30, 40) if d <= args.days]

    def trajectory(controller=None, start=initial):
        state, out, applied = start, {}, []
        for step in range(steps):
            tendency = None
            if controller is not None:
                t = controller.tendency()
                applied.append(t)
                tendency = None if not np.any(t) else {"temperature": t.transpose(0, 2, 1)}
            state = backend.advance_interval(state, backend.persisted_forcings(state, forcing), DT, tendency)
            if controller is not None:
                controller.observe_generated(temperature(state), applied[-1])
            if (step + 1) % 4 == 0 and (step + 1) // 4 in report_days:
                out[(step + 1) // 4] = temperature(state)
        return out, (np.stack(applied) if applied else None)

    t0 = time.time()
    base, _ = trajectory()
    print(f"baseline done in {time.time() - t0:.0f}s", flush=True)
    results = {}

    def summarize(name, run, applied=None):
        row = {}
        for day in report_days:
            diff = run[day] - base[day]
            row[str(day)] = {
                "rms_K": float(np.sqrt(np.sum(np.mean(diff**2, axis=0) * weights))),
                "global_mean_K": float(np.sum(np.mean(diff, axis=0) * weights)),
            }
        entry = {"by_day": row}
        if applied is not None:
            forced = applied[1:]
            entry["forcing_rms_K_per_day"] = float(np.sqrt(np.mean(forced**2)) * 86400)
            # Integrated forcing over the whole run: what a slow variable sees.
            entry["integrated_forcing_rms_K"] = float(np.sqrt(np.mean(forced.sum(0)**2)) * DT)
        results[name] = entry
        print(name, json.dumps(entry), flush=True)

    for name, spec in configs.items():
        checkpoint = make_checkpoint(sigma, gain=spec["gain"], slow=spec["slow"])
        controller = ClosureController(checkpoint, history, np.zeros_like(history), seed=11, mode="noise",
                                       correction_scale=1., tendency_clip_std=5., tile_columns=4096,
                                       noise_fn=spec["noise_fn"], memory_limit_gb=16,
                                       fast_noise_scale=spec.get("fast_scale", 1.))
        run, applied = trajectory(controller)
        summarize(name, run, applied)
    if args.only is None or "ic_tiny" in args.only:
        tiny = backend.perturb(initial, {"temperature": 0.01 * np.asarray(correlated(jax.random.PRNGKey(5), sigma)).swapaxes(-1, -2)})
        run, _ = trajectory(start=tiny)
        summarize("ic_tiny", run)
    with open(args.output, "w") as stream:
        json.dump({"days": report_days, "residual_std_K_per_s": RESIDUAL_STD, "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
