"""Does the closure noise inflate fast (sub-2-day) variability of the real TL63 core?

Companion to noise_response_tl63.py, with the same untrained closure and
prescribed amplitude. Temperature on sigma levels is sampled every native
step (1 h); its fast part is the deviation from a centred 48 h running mean.
Reported per configuration: area-weighted rms of that fast part over days
2..N, relative to the unforced baseline, and its slow (running-mean) part's
rms deviation from the baseline's.

Usage: python experiments/fast_band_tl63.py --days 6 --output runs/fast_band.json
"""
import argparse
import json

import jax
import numpy as np

from neuralgcm_mz.rollout import ClosureController
from noise_response_tl63 import DT, H, make_checkpoint


def main():
    import neuralgcm
    from neuralgcm import demo
    from neuralgcm_mz.neuralgcm_backend import NeuralGCMBackend
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = neuralgcm.PressureLevelModel.from_checkpoint(demo.load_checkpoint_tl63_stochastic())
    backend = NeuralGCMBackend(model)
    data = demo.load_data(model.data_coords).isel(time=0)
    forcing = model.forcings_from_xarray(data)
    initial = backend.encode(model.inputs_from_xarray(data), forcing, rng_key=jax.random.PRNGKey(0))
    sigma, lon, lat = backend.sigma_shape
    weights = np.cos(np.asarray(model.model_coords.horizontal.latitudes))[:, None] * np.ones((1, lon))
    weights = weights / weights.sum()
    temperature = lambda s: np.asarray(backend.sigma_fields(s)["temperature"]).transpose(0, 2, 1)
    history = np.repeat(temperature(initial)[None], H, axis=0)
    correlated = backend.spherical_noise(500)
    per_interval = round(DT / backend.timestep_seconds)

    def trajectory(controller=None):
        state, out = initial, []
        for _ in range(args.days * 4):
            tendency, t = None, None
            if controller is not None:
                t = controller.tendency()
                tendency = None if not np.any(t) else {"temperature": t.transpose(0, 2, 1)}
            for _ in range(per_interval):
                state = backend.step(state, backend.persisted_forcings(state, forcing), tendency)
                out.append(temperature(state).mean(axis=0))  # vertical mean, [lat, lon]
            if controller is not None:
                controller.observe_generated(temperature(state), t)
        return np.stack(out)

    def split(series, window=48):
        kernel = np.ones(window) / window
        slow = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, series)
        keep = slice(48, len(series) - window // 2)  # from day 2, away from the edges
        return (series - slow)[keep], slow[keep]

    wrms = lambda x: float(np.sqrt(np.sum(np.mean(x**2, axis=0) * weights)))
    base_fast, base_slow = split(trajectory())
    results = {"baseline": {"fast_rms_K": wrms(base_fast)}}
    configs = {
        "fast_white": dict(gain=1., slow=False, noise_fn=None),
        "fast_corr": dict(gain=1., slow=False, noise_fn=correlated),
        "slow_corr": dict(gain=0., slow=True, noise_fn=correlated),
        "full_corr": dict(gain=1., slow=True, noise_fn=correlated),
    }
    for name, spec in configs.items():
        controller = ClosureController(make_checkpoint(sigma, gain=spec["gain"], slow=spec["slow"]), history,
                                       np.zeros_like(history), seed=11, mode="noise", correction_scale=1.,
                                       tendency_clip_std=5., tile_columns=4096, noise_fn=spec["noise_fn"],
                                       memory_limit_gb=16)
        fast, slow = split(trajectory(controller))
        results[name] = {"fast_rms_K": wrms(fast), "fast_ratio_to_baseline": wrms(fast) / wrms(base_fast),
                         "slow_rms_diff_K": wrms(slow - base_slow)}
        print(name, json.dumps(results[name]), flush=True)
    with open(args.output, "w") as stream:
        json.dump({"days": args.days, "results": results}, stream, indent=2)


if __name__ == "__main__":
    main()
