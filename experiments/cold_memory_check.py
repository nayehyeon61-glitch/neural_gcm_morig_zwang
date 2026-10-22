"""How much of the training residual comes from NeuralGCM's cold memory?

prepare() encodes every interval afresh, so the 6 h teacher-forced step
starts with ModelState.memory equal to the current state (no past). Mid
forecast the memory is the state one native step earlier. For each sample
time t this script compares two 6 h steps from the same encoded state:

  cold    encode(t)                                  (as in the archive)
  warm    encode(t) with memory = model state at t - 1 step, obtained by
          integrating encode(t - 6 h) forward

Reported per variable: area-weighted rms of the cold residual, of the warm
residual, and of their difference (the cold-memory part), all in
residual units (per second), plus the hourly rms temperature tendency of the
cold run (an initial shock shows as a large first hour).

Usage: python experiments/cold_memory_check.py --checkpoint deterministic_2_8_deg.pkl \
           --era5 'data/*.nc' --times 2020-01-05T00 2020-01-09T00 --output cold_memory.json
"""
import argparse
import dataclasses
import json

import jax
import numpy as np

from neuralgcm_mz.neuralgcm_backend import NeuralGCMBackend
from neuralgcm_mz.rollout import pack_sigma
from neuralgcm_mz.workflow import ERA5Reader

VARIABLES = ("temperature", "u", "v", "specific_humidity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--era5", required=True)
    parser.add_argument("--times", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    backend = NeuralGCMBackend.from_checkpoint(args.checkpoint)
    reader = ERA5Reader(args.era5, backend)
    dt = 6 * 3600.
    native = round(dt / backend.timestep_seconds)
    sigma, lon, lat = backend.sigma_shape
    weights = np.cos(np.asarray(backend.model.model_coords.horizontal.latitudes))[None, :, None]
    weights = weights / weights.mean()
    key = jax.random.PRNGKey(0)

    def encoded(time):
        index = int(np.flatnonzero(reader.times == np.datetime64(time, "ns"))[0])
        _, inputs, forcing = reader.record(index)
        return backend.encode(inputs, forcing, rng_key=key), forcing

    def fields(state):
        return pack_sigma(backend.sigma_fields(state), VARIABLES)  # [C, lat, lon]

    def rms(x):
        x = x.reshape(len(VARIABLES), sigma, x.shape[-2], x.shape[-1])
        return {v: float(np.sqrt(np.mean(x[i]**2 * weights))) for i, v in enumerate(VARIABLES)}

    rows = []
    for time in args.times:
        t = np.datetime64(time, "ns")
        start, forcing = encoded(t)
        target = fields(encoded(t + np.timedelta64(6, "h"))[0])
        cold, hourly, previous = start, [], fields(start)
        for _ in range(native):
            cold = backend.step(cold, backend.persisted_forcings(cold, forcing), None)
            now = fields(cold)
            hourly.append(float(np.sqrt(np.mean((now - previous)[:sigma]**2 * weights)) / backend.timestep_seconds))
            previous = now
        prior, prior_forcing = encoded(t - np.timedelta64(6, "h"))
        for _ in range(native - 1):
            prior = backend.step(prior, backend.persisted_forcings(prior, prior_forcing), None)
        warm = dataclasses.replace(start, memory=prior.state)
        for _ in range(native):
            warm = backend.step(warm, backend.persisted_forcings(warm, forcing), None)
        r_cold = (target - fields(cold)) / dt
        r_warm = (target - fields(warm)) / dt
        row = {"time": time, "cold_rms": rms(r_cold), "warm_rms": rms(r_warm),
               "cold_minus_warm_rms": rms(r_cold - r_warm), "hourly_temperature_tendency_K_per_s": hourly}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {v: {"cold": float(np.mean([r["cold_rms"][v] for r in rows])),
                   "warm": float(np.mean([r["warm_rms"][v] for r in rows])),
                   "difference": float(np.mean([r["cold_minus_warm_rms"][v] for r in rows]))}
               for v in VARIABLES}
    with open(args.output, "w") as stream:
        json.dump({"samples": rows, "mean": summary}, stream, indent=2)


if __name__ == "__main__":
    main()
