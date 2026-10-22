"""Does generated noise reach future memory, and does it behave like the real residual?

Runs the closure alone, closed loop, over the validation split: the resolved
states come from the archive (true states), but every past correction the
memory sees is the closure's own output, as in a forecast. Two loops share
one seed:

  full     memory + fast DDPM + slow AR(1)   (what a forecast applies)
  memory   memory only                       (no generated noise)

(The injected noise of a block is the full loop's correction minus its own
memory mean, before the 5-sigma clip.)

For each block n the memory mean mu_n is recorded. The noise-to-memory
transfer is mu_n(full) - mu_n(memory): the part of the next block's memory
term that exists only because earlier blocks carried generated noise.
Reported per variable (normalized units, area weighted):

  std_ratio            rms of the generated correction / rms of the real residual
  lag_corr             autocorrelation at 6 h lags 1..2K of the generated
                       correction (full loop) and of the real residual
  boundary_corr        lag-1 correlation across block boundaries vs inside blocks
  transfer_rms         rms(mu_full - mu_memory) / rms(previous block noise), by lead
  transfer_beta        regression of the first-step transfer on the last-step noise
                       of the previous block (persistence carried by memory)
  block_rms            rms of the generated correction per block (drift check)

Usage: python experiments/closure_loop_check.py --archive runs/x/archive \
           --checkpoint runs/x/closure.npz --output runs/x/closure_loop.json
"""
import argparse
import json

import numpy as np

from neuralgcm_mz.data import ResidualArchive
from neuralgcm_mz.rollout import ClosureController
from neuralgcm_mz.training import load_checkpoint


def lag_corr(x, w, lags):
    """x [T, column, C] normalized; correlation at each lag pooled over columns/channels."""
    x = x - x.mean(axis=(0, 1))
    out = []
    for lag in lags:
        a, b = x[:-lag], x[lag:]
        out.append(float(np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))))
    return out


def excess_kurtosis(x):
    x = x - x.mean(axis=(0, 1))
    return float(np.mean(x**4) / np.mean(x**2)**2 - 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fast-noise-scale", type=float, default=0.5)
    parser.add_argument("--slow-noise-scale", type=float, default=1.0)
    parser.add_argument("--overlap-steps", type=int, default=0)
    parser.add_argument("--correlation-km", type=float, default=500.)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    archive = ResidualArchive(args.archive)
    checkpoint = load_checkpoint(args.checkpoint)
    config, norm = checkpoint["config"], checkpoint["normalizer"]
    h, k = config["history_steps"], config["block_steps"]
    first, last = archive.manifest["splits"]["validation"]
    origin = first + h            # history states origin-h+1..origin, residuals origin-h..origin-1
    stride = k - args.overlap_steps            # applied steps per block
    steps = (last - 1 - origin) // stride * stride
    _, c, ny, nx = archive.states.shape
    noise_fn = None
    if args.correlation_km > 0:
        from dinosaur import spherical_harmonic
        from neuralgcm_mz.spatial import spherical_noise
        grid = spherical_harmonic.Grid.TL63()
        if tuple(grid.nodal_shape) != (nx, ny):
            raise ValueError("Archive grid is not TL63; pass --correlation-km 0")
        noise_fn = spherical_noise(grid, args.correlation_km)

    def loop(mode):
        controller = ClosureController(
            checkpoint, np.asarray(archive.states[origin-h+1:origin+1]),
            np.asarray(archive.residual[origin-h:origin]), seed=args.seed, correction_scale=1.,
            tile_columns=2048, mode=mode, memory_limit_gb=12, noise_fn=noise_fn,
            fast_noise_scale=args.fast_noise_scale, slow_noise_scale=args.slow_noise_scale,
            overlap_steps=args.overlap_steps, issue_time=archive.timestamps[origin])
        means, recorder = [], controller._mean

        def recording(params, context):
            value = recorder(params, context)
            means.append(np.asarray(value))
            return value
        controller._mean = recording
        out = []
        for s in range(steps):
            x = controller.tendency()
            controller.observe_generated(np.asarray(archive.states[origin+s+1]), x)
            out.append(x)
        # Tiles are concatenated back per block; one block per k steps after the pure first step.
        tiles = -(-ny * nx // 2048)
        mu = np.stack([np.concatenate(means[i:i+tiles]) for i in range(0, len(means), tiles)])
        return np.stack(out), mu                    # [T, C, Y, X], [block, column, K, C]

    full, mu_full = loop("full")
    memory, mu_memory = loop("memory")
    to_columns = lambda x: ((x.transpose(0, 2, 3, 1).reshape(len(x), ny * nx, c)
                             - norm["residual_mean"]) / norm["residual_std"])
    generated = to_columns(full[1:])                # drop the pure-NeuralGCM first step
    mean_only = to_columns(memory[1:])
    real = to_columns(np.asarray(archive.residual[origin+1:origin+steps]))
    blocks = len(mu_full)
    # Injected noise: generated correction minus the memory mean of the same block.
    mu_sequence = mu_full[:, :, :stride].transpose(0, 2, 1, 3).reshape(blocks * stride, ny * nx, c)[:len(generated)]
    noise = generated - mu_sequence
    clip = 5.0                                      # controller default tendency_clip_std
    norm_center = np.zeros(c)                       # clip acts on normalized mean + noise
    weights = np.repeat(np.cos(np.deg2rad(archive.latitudes)), nx)
    weights = (weights / weights.mean())[None, :, None]
    names = [n.split(":")[0] for n in archive.channel_names]
    groups = {v: np.array([n == v for n in names]) for v in dict.fromkeys(names)}
    lags = list(range(1, 2 * k + 1))
    position = np.arange(len(generated)) % stride   # 0 = first step of a block
    report = {"checkpoint": args.checkpoint, "origin_time": str(archive.timestamps[origin]),
              "steps": steps, "blocks": blocks, "fast_noise_scale": args.fast_noise_scale,
              "slow_noise_scale": args.slow_noise_scale,
              "phase_features": bool(config.get("phase_features", False)),
              "max_period_hours": config["max_period_hours"], "overlap_steps": args.overlap_steps,
              "variables": {}}
    for var, sel in groups.items():
        g, r, e, m = generated[..., sel], real[..., sel], noise[..., sel], mean_only[..., sel]
        rms = lambda x: float(np.sqrt(np.mean(weights * x**2)))

        def pair_corr(x, starts):
            a = x[starts] - x.mean(axis=(0, 1)); b = x[starts + 1] - x.mean(axis=(0, 1))
            return float(np.sum(weights * a * b) / np.sqrt(np.sum(weights * a * a) * np.sum(weights * b * b)))
        inside = np.flatnonzero(position[:-1] != stride - 1)
        across = np.flatnonzero(position[:-1] == stride - 1)
        transfer = mu_full[1:, :, :stride][..., sel] - mu_memory[1:, :, :stride][..., sel]  # [block-1, column, stride, C]
        previous = np.stack([e[(b - 1) * stride:b * stride] for b in range(1, blocks)])  # noise of block b-1
        transfer_rms = [float(np.sqrt(np.mean(weights[0] * transfer[:, :, j]**2)) / rms(previous))
                        for j in range(stride)]
        a, b = transfer[:, :, 0], previous[:, -1]
        beta = float(np.sum(weights[0] * a * b) / np.sum(weights[0] * b * b))
        corr = float(np.sum(weights[0] * a * b) / np.sqrt(np.sum(weights[0] * a * a) * np.sum(weights[0] * b * b)))
        report["variables"][var] = {
            "std_ratio": rms(g - g.mean(axis=(0, 1))) / rms(r - r.mean(axis=(0, 1))),
            "noise_std_ratio": rms(e) / rms(r - r.mean(axis=(0, 1))),
            "memory_only_std_ratio": rms(m - m.mean(axis=(0, 1))) / rms(r - r.mean(axis=(0, 1))),
            "lag_corr_generated": lag_corr(g, weights, lags), "lag_corr_real": lag_corr(r, weights, lags),
            "boundary_corr": {"generated_inside": pair_corr(g, inside), "generated_across": pair_corr(g, across),
                              "real_inside": pair_corr(r, inside), "real_across": pair_corr(r, across)},
            "transfer_rms_by_lead": transfer_rms, "transfer_beta": beta, "transfer_corr": corr,
            "block_rms": [rms(g[i*stride:(i+1)*stride]) for i in range(len(g) // stride)],
            "kurtosis": {"generated": excess_kurtosis(g), "real": excess_kurtosis(r)},
            # Share of entries at the 5-sigma tendency clip (normalized units).
            "clipped_fraction": float(np.mean(np.abs(g - norm_center[sel]) >= 0.999 * clip)),
        }
        print(var, json.dumps(report["variables"][var]), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
