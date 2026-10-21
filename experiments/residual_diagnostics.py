"""What the closure's noise is made of, measured on a real residual archive.

Uses validation windows only. For every window, target = normalized future
residual block, mu = closure mean, e = target - mu (the DDPM/AR(1) input).
Reported per variable (temperature, u, v, specific_humidity):

  r2_mu                 share of residual variance the memory term explains
  corr_e_mu             correlation of the innovation with the memory term
  r2_e_from_local_h     out-of-sample R^2 of e (first step) from the local
                        latest state and residual (linear, per channel)
  mode_energy           share of e variance in DCT mode 0 (block mean), 1, >=2
  lag_corr              correlation of e at lags 1..K-1 inside a block (6 h steps)
  block_phi             lag-1 correlation of consecutive block means
  skewness/kurtosis     of e, of its fast (k>=2) part and of its block mean
  zonal_length_km       correlation length of the fast part and the block mean
  memory_response       rms change of mu per rms change of past residuals when
                        innovation-sized noise is added to them, by lead in the block

Usage: python experiments/residual_diagnostics.py --archive runs/x/archive \
           --checkpoint runs/x/closure.npz --output runs/x/residual_diagnostics.json
"""
import argparse
import json

import jax
import numpy as np

from neuralgcm_mz.closure import predict_closure
from neuralgcm_mz.data import ResidualArchive
from neuralgcm_mz.spectral import dct, highpass_mask, idct
from neuralgcm_mz.training import _window, load_checkpoint, zonal_correlation_length


def moments(x):
    x = x - x.mean()
    s = x.std()
    return float(np.mean(x**3) / s**3), float(np.mean(x**4) / s**4 - 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive = ResidualArchive(args.archive)
    checkpoint = load_checkpoint(args.checkpoint)
    config, normalizer = checkpoint["config"], checkpoint["normalizer"]
    h, k = config["history_steps"], config["block_steps"]
    mask = np.asarray(highpass_mask(k, config["dt_hours"], config["max_period_hours"]))
    params = checkpoint["closure_params"]
    mean_fn = jax.jit(lambda hist: predict_closure(params, hist))
    origins = archive.window_origins("validation", h, k)
    # Consecutive, non-overlapping blocks: every window has a successor for block_phi.
    origins = origins[::k]
    c, ny, nx = archive.states.shape[1:]
    ys, xs = np.divmod(np.arange(ny * nx), nx)
    weights = np.repeat(np.cos(np.deg2rad(archive.latitudes)), nx)
    weights = weights / weights.mean()
    names = [n.split(":")[0] for n in archive.channel_names]
    groups = {v: np.array([n == v for n in names]) for v in dict.fromkeys(names)}
    rng = np.random.default_rng(0)

    class Pool(dict):
        def add(self, key, value):
            self.setdefault(key, []).append(value)

    pool = Pool()
    blocks = {}
    for origin in origins:
        hist, target = _window(archive, origin, ys, xs, type("C", (), {"history_steps": h, "block_steps": k}),
                               normalizer)
        mu = np.asarray(mean_fn(hist))
        e = target - mu                                   # [column, K, C]
        coeff = np.asarray(dct(e, axis=1))
        fast = np.asarray(idct(coeff * mask[None, :, None], axis=1))
        blocks[origin] = e.mean(axis=1)
        # Memory response: perturb past residuals by innovation-sized noise.
        delta = rng.standard_normal(hist[..., c:].shape).astype(np.float32) * e.std(axis=(0, 1))[None, None]
        perturbed = hist.copy()
        perturbed[..., c:] += delta
        dmu = np.asarray(mean_fn(perturbed)) - mu
        pool.add("target", target); pool.add("mu", mu); pool.add("e", e); pool.add("coeff", coeff)
        pool.add("fast", fast); pool.add("dmu", dmu); pool.add("delta_rms", np.sqrt(np.mean(delta**2, axis=(0, 1))))
        pool.add("local", np.stack([hist[:, -1, :c], hist[:, -1, c:]], axis=-1))  # latest state, latest residual
    t, mu, e = (np.stack(pool[n]) for n in ("target", "mu", "e"))      # [window, column, K, C]
    coeff, fast, dmu, local = (np.stack(pool[n]) for n in ("coeff", "fast", "dmu", "local"))
    delta_rms = np.mean(pool["delta_rms"], axis=0)
    w = weights[None, :, None]
    phis = {}
    pairs = [(o, o + k) for o in blocks if o + k in blocks]
    report = {"windows": len(origins), "block_pairs": len(pairs), "variables": {}}
    for var, sel in groups.items():
        tv, mv, ev = t[..., sel], mu[..., sel], e[..., sel]
        wv = w[..., None]
        r2 = 1 - np.sum(wv * ev**2) / np.sum(wv * (tv - tv.mean(axis=(0, 1, 2)))**2)
        corr_e_mu = np.sum(wv * ev * mv) / np.sqrt(np.sum(wv * ev**2) * np.sum(wv * mv**2))
        # Linear predictability of the first-step innovation from local history, split-half.
        half = len(origins) // 2
        scores = []
        for ch in np.flatnonzero(sel):
            X = np.concatenate([local[..., ch, :], np.ones(local.shape[:2] + (1,))], -1)
            yv = e[:, :, 0, ch]
            beta, *_ = np.linalg.lstsq(X[:half].reshape(-1, 3), yv[:half].reshape(-1), rcond=None)
            pred = X[half:].reshape(-1, 3) @ beta
            yt = yv[half:].reshape(-1)
            scores.append(1 - np.mean((yt - pred)**2) / np.var(yt))
        energy = np.sum(coeff[..., sel]**2 * wv, axis=(0, 1, 3))
        energy = energy / energy.sum()
        lags = [float(np.sum(wv[:, :, :1] * ev[:, :, :-l] * ev[:, :, l:]) /
                      np.sqrt(np.sum(wv[:, :, :1] * ev[:, :, :-l]**2) * np.sum(wv[:, :, :1] * ev[:, :, l:]**2)))
                for l in range(1, k)]
        if pairs:
            a = np.stack([blocks[p][:, sel] for p, _ in pairs]); b = np.stack([blocks[q][:, sel] for _, q in pairs])
            a, b = a - a.mean(axis=(0, 1)), b - b.mean(axis=(0, 1))
            phi = float(np.sum(a * b) / np.sum(a * a))
        else:
            phi = None
        mean_block = ev.mean(axis=2)
        fv = fast[..., sel]
        fields = lambda x: x.reshape(x.shape[0], ny, nx, -1)
        resp = np.sqrt(np.mean(dmu[..., sel]**2, axis=(0, 1, 3))) / float(np.mean(delta_rms[sel]))
        report["variables"][var] = {
            "r2_mu": float(r2), "corr_e_mu": float(corr_e_mu),
            "r2_e_from_local_h_median": float(np.median(scores)),
            "mode_energy": {"k0_block_mean": float(energy[0]), "k1_96h": float(energy[1]),
                            "k2plus_fast": float(energy[mask > 0].sum())},
            "lag_corr_6h_steps": lags, "block_phi": phi,
            "skew_kurtosis": {"e": moments(ev.ravel()), "fast": moments(fv.ravel()),
                              "block_mean": moments(mean_block.ravel())},
            "zonal_length_km": {"fast": zonal_correlation_length(fields(fv[:, :, 0]), archive.latitudes),
                                "block_mean": zonal_correlation_length(fields(mean_block), archive.latitudes)},
            "memory_response_by_lead": [float(x) for x in resp],
            "innovation_std_normalized": float(ev.std()),
        }
        print(var, json.dumps(report["variables"][var]), flush=True)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
