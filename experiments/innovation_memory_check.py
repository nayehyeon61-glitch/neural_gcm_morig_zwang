"""Is there memory left in the innovation that the closure throws away?

Theory (Gottwald et al. 2016; Chorin & Lu 2015): the memory chain may be cut
where the remainder is white. The closure's remainder is the innovation
e = target - mu. If e still carries memory, a NARMAX-type term (past
innovations) or a recurrent hidden state could use it; if not, a longer
memory cannot help and the effort belongs elsewhere.

Three checks, no forecasts (archive + closure checkpoint only):

  M1 whiteness   one-step innovation e_n = r_n - mu(h_n)[0] at every 6 h step
                 of a split: pooled autocorrelation at lags 1..16 (6-96 h) per
                 variable, and the share of column-channels whose Ljung-Box
                 Q(16) rejects whiteness at the 1% level.
  M2 NARMAX bound  how much of the NEXT block's innovation (leads 0..K-1) is
                 linearly predictable from the p innovations before the block
                 (MA/AR-on-noise part), and from those plus the p residuals
                 (z part). Per-channel coefficients shared by all columns are
                 fitted on train and scored out of sample on validation (R^2).
  M3 lag importance  increase of the mean-closure MSE when one history lag
                 (state and residual of that step) is shuffled across columns.

Usage: python experiments/innovation_memory_check.py --archive runs/x/archive \
           --checkpoint runs/x/closure.npz --output runs/x/innovation_memory.json
"""
import argparse
import json

import jax
import numpy as np
from scipy import stats

from neuralgcm_mz.closure import predict_closure
from neuralgcm_mz.data import ResidualArchive
from neuralgcm_mz.training import _window, load_checkpoint


def innovations(archive, split, h, k, mean_fn, ys, xs, window_config, normalizer):
    """One-step innovations, residuals and per-lead block innovations at every step."""
    origins = archive.window_origins(split, h, k)
    e1, r1, blocks = [], [], {}
    for origin in origins:
        hist, target = _window(archive, origin, ys, xs, window_config, normalizer)
        e = target - np.asarray(mean_fn(hist))           # [column, K, C]
        e1.append(e[:, 0]); r1.append(target[:, 0]); blocks[origin] = e
    return list(origins), np.stack(e1), np.stack(r1), blocks


def pooled_acf(x, w, lags):
    x = x - x.mean(axis=0)
    denom = np.sum(w * x * x)
    return [float(np.sum(w * x[:-l] * x[l:]) / denom * len(x) / (len(x) - l)) for l in lags]


def ljung_box_reject(x, lags=16, level=0.01):
    """Share of series (columns x channels) whose Q(lags) rejects whiteness."""
    t = len(x)
    x = x - x.mean(axis=0)
    var = np.sum(x * x, axis=0)
    q = np.zeros(x.shape[1:])
    for l in range(1, lags + 1):
        rho = np.sum(x[:-l] * x[l:], axis=0) / np.maximum(var, 1e-12)
        q += rho**2 / (t - l)
    q *= t * (t + 2)
    return float(np.mean(q > stats.chi2.ppf(1 - level, lags)))


def design(e1, r1, starts, p, use_r):
    """Rows: block start s; features: e (and r) at s-1..s-p, per column and channel."""
    feats = [e1[[s - j for s in starts]] for j in range(1, p + 1)]
    if use_r:
        feats += [r1[[s - j for s in starts]] for j in range(1, p + 1)]
    return np.stack(feats, axis=-1)                     # [block, column, C, F]


def fit_per_channel(x, y):
    """Least squares per channel, pooled over blocks and columns. x [B,N,C,F], y [B,N,C]."""
    c, f = x.shape[2], x.shape[3]
    coef = np.zeros((c, f))
    for i in range(c):
        a = x[:, :, i].reshape(-1, f); b = y[:, :, i].reshape(-1)
        coef[i] = np.linalg.lstsq(a.T @ a + 1e-3 * np.eye(f), a.T @ b, rcond=None)[0]
    return coef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--column-stride", type=int, default=8)
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    archive = ResidualArchive(args.archive)
    checkpoint = load_checkpoint(args.checkpoint)
    config, normalizer = checkpoint["config"], checkpoint["normalizer"]
    h, k = config["history_steps"], config["block_steps"]
    params = checkpoint["closure_params"]
    mean_fn = jax.jit(lambda hist: predict_closure(params, hist))
    window_config = type("C", (), {"history_steps": h, "block_steps": k,
                                   "phase_features": config.get("phase_features", False)})
    c, ny, nx = archive.states.shape[1:]
    columns = np.arange(0, ny * nx, args.column_stride)
    ys, xs = np.divmod(columns, nx)
    w = np.cos(np.deg2rad(archive.latitudes[ys])); w = (w / w.mean())[None, :, None]
    names = [n.split(":")[0] for n in archive.channel_names]
    groups = {v: np.array([n == v for n in names]) for v in dict.fromkeys(names)}
    data = {s: innovations(archive, s, h, k, mean_fn, ys, xs, window_config, normalizer)
            for s in ("train", "validation")}
    report = {"checkpoint": args.checkpoint, "columns": len(columns), "variables": {}}

    # M2 design: block starts whose p previous steps and K future leads all exist.
    def starts(split, p):
        origins, e1, _, blocks = data[split]
        idx = [i for i in range(p, len(origins)) if origins[i] in blocks]
        return idx
    m2 = {}
    for p in args.orders:
        for use_r in (False, True):
            key = f"p{p}_{'e+r' if use_r else 'e'}"
            fits = {}
            for split in ("train", "validation"):
                origins, e1, r1, blocks = data[split]
                idx = starts(split, p)
                x = design(e1, r1, idx, p, use_r)
                y = np.stack([blocks[origins[i]] for i in idx])     # [B, N, K, C]
                fits[split] = (x, y)
            x_tr, y_tr = fits["train"]; x_va, y_va = fits["validation"]
            r2 = {v: [] for v in groups}
            for lead in range(k):
                coef = fit_per_channel(x_tr, y_tr[:, :, lead])
                pred = np.einsum("bncf,cf->bnc", x_va, coef)
                err = y_va[:, :, lead] - pred
                for v, sel in groups.items():
                    tot = np.sum(w * y_va[:, :, lead][..., sel]**2)
                    r2[v].append(float(1 - np.sum(w * err[..., sel]**2) / tot))
            m2[key] = r2
            print("M2", key, {v: round(np.mean(r), 4) for v, r in r2.items()}, flush=True)

    # M3: shuffle one history lag across columns.
    origins = archive.window_origins("validation", h, k)[::k]
    rng = np.random.default_rng(0)
    base, lag_mse = [], {l: [] for l in range(h)}
    for origin in origins:
        hist, target = _window(archive, origin, ys, xs, window_config, normalizer)
        base.append(float(np.mean((target - np.asarray(mean_fn(hist)))**2)))
        for l in range(h):                              # index h-1 = latest step
            shuffled = hist.copy()
            shuffled[:, l] = hist[rng.permutation(len(hist)), l]
            lag_mse[l].append(float(np.mean((target - np.asarray(mean_fn(shuffled)))**2)))
    base_mse = float(np.mean(base))
    m3 = {f"lag_{(h - 1 - l) * int(config['dt_hours'])}h": float(np.mean(v) / base_mse - 1)
          for l, v in lag_mse.items()}
    print("M3", m3, flush=True)

    _, e_va, _, _ = data["validation"]
    lags = list(range(1, 17))
    for v, sel in groups.items():
        x = e_va[..., sel]
        report["variables"][v] = {
            "M1_acf_lags_6h": pooled_acf(x, w, lags),
            "M1_ljung_box_reject_share": ljung_box_reject(x.reshape(len(x), -1)),
            "M2_r2_by_lead": {key: r2[v] for key, r2 in m2.items()},
        }
        print(v, json.dumps({"acf": [round(a, 3) for a in report["variables"][v]["M1_acf_lags_6h"][:8]],
                             "lb": report["variables"][v]["M1_ljung_box_reject_share"]}), flush=True)
    report["M3_relative_mse_increase_by_lag"] = m3
    report["M3_base_mse"] = base_mse
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
