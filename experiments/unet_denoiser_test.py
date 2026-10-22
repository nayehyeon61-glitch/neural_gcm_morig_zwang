"""Column MLP vs spherical U-Net as the DDPM denoiser, on a real residual archive.

The repository's denoiser is a per-column MLP: every grid column is denoised
independently and spatial structure only enters through correlated Gaussian
draws. This experiment asks whether a spatial U-Net backbone (all columns at
once, multi-scale convolutions on the sphere) generates more realistic noise.

All models generate the same target: the time-DCT fast coefficients (modes
allowed by the checkpoint mask) of the innovation e = target - mu, with mu
from the trained closure checkpoint. Variants:

  current     checkpoint DDPM (column MLP, full 8-step history), correlated draws
  mlp         column MLP, light conditioning, trained here, white draws
  *_corr      a trained variant sampled with correlated draws (checkpoint length)
  unet        spherical U-Net, light conditioning, trained here, white draws
  unet_attn   U-Net with global self-attention at the coarsest level
  gaussian    per-(mode, channel) Gaussian, spatially white (reference)
  base        exact Gaussian denoiser per (mode, channel), no network
  unet3d      (2+1)-D U-Net: vertical levels as their own axis (shared 2-D
              convolutions per level, 1-D convolutions across levels)
  *_skip      mlp / unet / unet_attn learning a residual on top of base

Light conditioning, identical for mlp and unet: latest normalized state,
latest normalized residual, sin/cos latitude. The comparison isolates the
backbone. Each sampled variant gets a per-(mode, channel) amplitude gain fitted
on half the validation windows (as training does); scores use the other half.

Scores (normalized units, area weighted, fast part in the time domain):
  eps_mse              validation denoising loss (not for current/gaussian)
  crps                 fair 4-member CRPS of the coefficients
  neighbour_corr       zonal and meridional lag-1 correlation, per variable
  zonal_power_ratio    generated / real zonal power in wavenumbers 1-4, 5-16, 17-64
  vertical_corr        adjacent-level correlation
  tq_corr              same-point temperature-humidity correlation
  kurtosis, lag1_time  per variable

Options: --corr-km sets the injected correlation length of *_corr variants;
--x0-clip the sampler's clean-sample clip (inf disables it); --transform
asinh trains and samples in y = s asinh(x / s) per channel, with s chosen so
that the training y has no excess kurtosis, and maps samples back with sinh
(heavy tails from a Gaussian-like diffusion target).
Multivariate scores: variogram score (p = 0.5) over zonal, meridional,
vertical and 6 h pairs, and the fair energy score of each field.

Usage: python experiments/unet_denoiser_test.py --archive runs/x/archive \
           --checkpoint runs/x/closure.npz --work runs/x/unet_work --output runs/x/unet_test.json
"""
import argparse
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from neuralgcm_mz.closure import predict_closure
from neuralgcm_mz.data import ResidualArchive
from neuralgcm_mz.diffusion import X0_CLIP, denoise, schedule
from neuralgcm_mz.spectral import dct, highpass_mask, idct
from neuralgcm_mz.training import _window, load_checkpoint

STEPS = 32  # diffusion steps, as in the checkpoints


# ---------------------------------------------------------------- data

def build_dataset(archive, checkpoint, work, train_stride=1):
    """Full-grid conditioning and target coefficients, float16 memmaps on disk."""
    config, norm = checkpoint["config"], checkpoint["normalizer"]
    h, k = config["history_steps"], config["block_steps"]
    mask = np.asarray(highpass_mask(k, config["dt_hours"], config["max_period_hours"]))
    _, c, ny, nx = archive.states.shape
    ys, xs = np.divmod(np.arange(ny * nx), nx)
    window_config = SimpleNamespace(history_steps=h, block_steps=k,
                                    phase_features=config.get("phase_features", False))
    mu_fn = jax.jit(lambda hist: predict_closure(checkpoint["closure_params"], hist))
    lat = np.deg2rad(archive.latitudes)
    geo = np.stack([np.repeat(np.sin(lat), nx), np.repeat(np.cos(lat), nx)], -1).astype(np.float32)
    out = {}
    for split in ("train", "validation"):
        origins = list(archive.window_origins(split, h, k))
        origins = origins[::4] if split == "validation" else origins[::train_stride]
        if (work / f"{split}.done").exists():  # reuse a finished build
            out[split] = (np.load(work / f"{split}_cond.npy", mmap_mode="r"),
                          np.load(work / f"{split}_coeff.npy", mmap_mode="r"))
            continue
        cond = np.lib.format.open_memmap(work / f"{split}_cond.npy", "w+", np.float16, (len(origins), ny * nx, 2 * c + 2))
        coeff = np.lib.format.open_memmap(work / f"{split}_coeff.npy", "w+", np.float16, (len(origins), ny * nx, k, c))
        full = []  # full histories of validation windows, for the checkpoint DDPM
        for i, origin in enumerate(origins):
            hist, target = _window(archive, origin, ys, xs, window_config, norm)
            e = target - np.asarray(mu_fn(hist))
            coeff[i] = np.asarray(dct(e, axis=1)) * mask[None, :, None]
            cond[i] = np.concatenate([hist[:, -1, :c], hist[:, -1, c:2 * c], geo], -1)
            if split == "validation":
                full.append(hist.astype(np.float16))
        cond.flush(); coeff.flush()
        (work / f"{split}.done").touch()
        out[split] = (cond, coeff)
        if split == "validation":
            np.save(work / "validation_history.npy", np.stack(full))
        print(f"{split}: {len(origins)} windows", flush=True)
    return out, mask, (ny, nx, k, c)


# ---------------------------------------------------------------- models

def dense(key, n_in, n_out, scale=1.0):
    return {"w": jax.random.normal(key, (n_in, n_out)) * scale / np.sqrt(n_in), "b": jnp.zeros(n_out)}


def apply_dense(p, x):
    return x @ p["w"] + p["b"]


def embed(t):
    t = t[:, None]
    return jnp.concatenate((t, t * t, jnp.sin(jnp.pi * t), jnp.cos(jnp.pi * t)), -1)


def init_mlp(key, n_in, n_out, hidden=128):
    a, b, c = jax.random.split(key, 3)
    return {"l1": dense(a, n_in + 4, hidden), "l2": dense(b, hidden, hidden), "l3": dense(c, hidden, n_out, 0.01)}


def mlp(params, noisy, cond, t):
    """Per column: noisy [B,N,F], cond [B,N,G], t [B] -> eps [B,N,F]."""
    e = jnp.broadcast_to(embed(t)[:, None], noisy.shape[:2] + (4,))
    x = jax.nn.silu(apply_dense(params["l1"], jnp.concatenate((noisy, cond, e), -1)))
    x = jax.nn.silu(apply_dense(params["l2"], x))
    return apply_dense(params["l3"], x)


def sphere_pad(x):
    """[B,Y,X,C] -> [B,Y+2,X+2,C]: periodic in longitude, across the poles in latitude."""
    half = x.shape[2] // 2
    x = jnp.concatenate((jnp.roll(x[:, :1], half, 2), x, jnp.roll(x[:, -1:], half, 2)), 1)
    return jnp.concatenate((x[:, :, -1:], x, x[:, :, :1]), 2)


def conv3(p, x):
    y = jax.lax.conv_general_dilated(sphere_pad(x), p["w"], (1, 1), "VALID",
                                     dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + p["b"]


def init_conv3(key, n_in, n_out, scale=1.0):
    return {"w": jax.random.normal(key, (3, 3, n_in, n_out)) * scale / np.sqrt(9 * n_in), "b": jnp.zeros(n_out)}


def rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)


def init_res(key, ch):
    a, b, c = jax.random.split(key, 3)
    return {"c1": init_conv3(a, ch, ch), "c2": init_conv3(b, ch, ch, 0.1), "t": dense(c, 4, ch)}


def res(p, x, temb):
    h = conv3(p["c1"], jax.nn.silu(rms_norm(x))) + apply_dense(p["t"], temb)[:, None, None]
    return x + conv3(p["c2"], jax.nn.silu(rms_norm(h)))


def init_unet(key, n_in, n_out, widths=(96, 128, 192), attention=False):
    keys = iter(jax.random.split(key, 16))
    w1, w2, w3 = widths
    p = {"stem": dense(next(keys), n_in, w1), "r1": init_res(next(keys), w1),
         "d2": dense(next(keys), w1, w2), "r2": init_res(next(keys), w2),
         "d3": dense(next(keys), w2, w3), "r3": init_res(next(keys), w3), "r3b": init_res(next(keys), w3),
         "u2": dense(next(keys), w3 + w2, w2), "r2u": init_res(next(keys), w2),
         "u1": dense(next(keys), w2 + w1, w1), "r1u": init_res(next(keys), w1),
         "head": dense(next(keys), w1, n_out, 0.01)}
    if attention:
        p["attn"] = {"q": dense(next(keys), w3, w3), "k": dense(next(keys), w3, w3),
                     "v": dense(next(keys), w3, w3), "o": dense(next(keys), w3, w3, 0.1)}
    return p


def attend(p, x, heads=4):
    b, y, xx, ch = x.shape
    z = rms_norm(x).reshape(b, y * xx, ch)
    split = lambda a: a.reshape(b, y * xx, heads, ch // heads)
    q, k, v = (split(apply_dense(p[n], z)) for n in ("q", "k", "v"))
    w = jax.nn.softmax(jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(ch // heads), -1)
    o = jnp.einsum("bhqk,bkhd->bqhd", w, v).reshape(b, y * xx, ch)
    return x + apply_dense(p["o"], o).reshape(b, y, xx, ch)


def pool(x):
    b, y, xx, c = x.shape
    return x.reshape(b, y // 2, 2, xx // 2, 2, c).mean((2, 4))


def up(x):
    return jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)


def unet(params, noisy, cond, t, grid):
    ny, nx = grid
    temb = embed(t)
    x = jnp.concatenate((noisy, cond), -1).reshape(noisy.shape[0], ny, nx, -1)
    a = res(params["r1"], apply_dense(params["stem"], x), temb)
    b = res(params["r2"], apply_dense(params["d2"], pool(a)), temb)
    c = res(params["r3"], apply_dense(params["d3"], pool(b)), temb)
    if "attn" in params:
        c = attend(params["attn"], c)
    c = res(params["r3b"], c, temb)
    b = res(params["r2u"], apply_dense(params["u2"], jnp.concatenate((up(c), b), -1)), temb)
    a = res(params["r1u"], apply_dense(params["u1"], jnp.concatenate((up(b), a), -1)), temb)
    return apply_dense(params["head"], a).reshape(noisy.shape)


def init_level_conv(key, n_in, n_out, scale=1.0):
    return {"w": jax.random.normal(key, (3, n_in, n_out)) * scale / np.sqrt(3 * n_in), "b": jnp.zeros(n_out)}


def level_conv(p, x):
    """1-D convolution across levels, axis -2 of [B,Y,X,L,C]; edge-replicated."""
    padded = jnp.concatenate((x[..., :1, :], x, x[..., -1:, :]), -2)
    return sum(jnp.einsum("byxlc,cd->byxld", padded[..., i:i + x.shape[-2], :], p["w"][i]) for i in range(3)) + p["b"]


def conv3_levels(p, x):
    """Spatial 3x3 on the sphere, shared across levels."""
    b, y, xx, l, c = x.shape
    z = x.transpose(0, 3, 1, 2, 4).reshape(b * l, y, xx, c)
    z = conv3(p, z)
    return z.reshape(b, l, y, xx, -1).transpose(0, 2, 3, 1, 4)


def init_res3(key, ch):
    a, b, c, d = jax.random.split(key, 4)
    return {"s": init_conv3(a, ch, ch), "v": init_level_conv(b, ch, ch), "s2": init_conv3(c, ch, ch, 0.1),
            "t": dense(d, 4, ch)}


def res3(p, x, temb):
    h = conv3_levels(p["s"], jax.nn.silu(rms_norm(x)))
    h = level_conv(p["v"], jax.nn.silu(rms_norm(h))) + apply_dense(p["t"], temb)[:, None, None, None]
    return x + conv3_levels(p["s2"], jax.nn.silu(rms_norm(h)))


def pool3(x):
    b, y, xx, l, c = x.shape
    return x.reshape(b, y // 2, 2, xx // 2, 2, l, c).mean((2, 4))


def up3(x):
    return jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)


def init_unet3d(key, n_in, n_out, widths=(24, 32, 48)):
    keys = iter(jax.random.split(key, 12))
    w1, w2, w3 = widths
    return {"stem": dense(next(keys), n_in, w1), "r1": init_res3(next(keys), w1),
            "d2": dense(next(keys), w1, w2), "r2": init_res3(next(keys), w2),
            "d3": dense(next(keys), w2, w3), "r3": init_res3(next(keys), w3),
            "u2": dense(next(keys), w3 + w2, w2), "r2u": init_res3(next(keys), w2),
            "u1": dense(next(keys), w2 + w1, w1), "r1u": init_res3(next(keys), w1),
            "head": dense(next(keys), w1, n_out, 0.01)}


def unet3d(params, noisy, cond, t, grid, k, variables, levels):
    """noisy [B,N,K*C] with C ordered variable-major; cond [B,N,2C+2]."""
    ny, nx = grid
    b = noisy.shape[0]
    temb = embed(t)
    x = noisy.reshape(b, ny, nx, k, variables, levels).transpose(0, 1, 2, 5, 3, 4).reshape(b, ny, nx, levels, k * variables)
    c = variables * levels
    per_level = lambda z: z.reshape(b, ny, nx, variables, levels).transpose(0, 1, 2, 4, 3)
    geo = jnp.broadcast_to(cond[..., 2 * c:].reshape(b, ny, nx, 1, -1), (b, ny, nx, levels, cond.shape[-1] - 2 * c))
    x = jnp.concatenate((x, per_level(cond[..., :c]), per_level(cond[..., c:2 * c]), geo), -1)
    a = res3(params["r1"], apply_dense(params["stem"], x), temb)
    m = res3(params["r2"], apply_dense(params["d2"], pool3(a)), temb)
    z = res3(params["r3"], apply_dense(params["d3"], pool3(m)), temb)
    m = res3(params["r2u"], apply_dense(params["u2"], jnp.concatenate((up3(z), m), -1)), temb)
    a = res3(params["r1u"], apply_dense(params["u1"], jnp.concatenate((up3(m), a), -1)), temb)
    out = apply_dense(params["head"], a)                       # [B,Y,X,L,K*V]
    out = out.reshape(b, ny, nx, levels, k, variables).transpose(0, 1, 2, 4, 5, 3)
    return out.reshape(noisy.shape)


# ---------------------------------------------------------------- diffusion

def loss_fn(model, params, key, x0, cond, m):
    tk, nk = jax.random.split(key)
    t = jax.random.randint(tk, (x0.shape[0],), 0, STEPS)
    _, _, abar = schedule(STEPS)
    noise = jax.random.normal(nk, x0.shape) * m
    a = abar[t][:, None, None]
    pred = model(params, jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * noise, cond, t / (STEPS - 1)) * m
    return jnp.sum((pred - noise) ** 2) / (jnp.sum(m) * x0.shape[0] * x0.shape[1])


def sample(model, params, key, cond, m, shape, draw, clip=X0_CLIP):
    """Ancestral sampling with the repository's x0-clipped posterior mean."""
    beta, alpha, abar = schedule(STEPS)
    keys = jax.random.split(key, STEPS + 1)
    x = draw(keys[0], shape) * m
    for i in range(STEPS):
        t = STEPS - 1 - i
        eps = model(params, x, cond, jnp.full((shape[0],), t / (STEPS - 1))) * m
        prev = abar[t - 1] if t > 0 else 1.0
        x0 = jnp.clip((x - jnp.sqrt(1 - abar[t]) * eps) / jnp.sqrt(abar[t]), -clip, clip)
        mean = (jnp.sqrt(prev) * beta[t] * x0 + jnp.sqrt(alpha[t]) * (1 - prev) * x) / (1 - abar[t])
        var = beta[t] * (1 - prev) / (1 - abar[t])
        x = (mean + (jnp.sqrt(var) * draw(keys[i + 1], shape) if t > 0 else 0)) * m
    return x


def train(name, model, params, data, m, steps, lr, batch, seed):
    cond, coeff = data["train"]
    vcond, vcoeff = data["validation"]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = opt.init(params)

    @jax.jit
    def update(params, state, key, x0, c):
        loss, grad = jax.value_and_grad(lambda p: loss_fn(model, p, key, x0, c, m))(params)
        updates, state = opt.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss

    evaluate = jax.jit(lambda p, key, x0, c: loss_fn(model, p, key, x0, c, m))
    vkeys = jax.random.split(jax.random.PRNGKey(seed + 1), len(vcoeff))

    def validation(p):
        return float(np.mean([float(evaluate(p, vkeys[i], jnp.asarray(vcoeff[i:i+1], jnp.float32),
                                             jnp.asarray(vcond[i:i+1], jnp.float32)))
                              for i in range(len(vcoeff))]))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    best, best_params, trace, start = np.inf, params, [], time.time()
    for step in range(1, steps + 1):
        idx = np.sort(rng.choice(len(coeff), batch, replace=False))
        key, sub = jax.random.split(key)
        params, state, loss = update(params, state, sub, jnp.asarray(coeff[idx], jnp.float32),
                                     jnp.asarray(cond[idx], jnp.float32))
        if step % 250 == 0 or step == steps:
            value = validation(params)
            trace.append({"step": step, "train": float(loss), "validation": value, "seconds": time.time() - start})
            print(name, trace[-1], flush=True)
            if value < best:
                best, best_params = value, params
    return best_params, best, trace


# ---------------------------------------------------------------- scores

def scores(gen, real, weights, names, mask):
    """gen [S,W,N,K,C] coefficients (S members), real [W,N,K,C]; returns a dict."""
    ny, nx = weights["grid"]
    w = weights["w"]                                         # [N]
    member = gen[0].astype(np.float32)
    # Fair CRPS, accumulated pairwise to keep memory at a few member-sized arrays.
    m = len(gen)
    ens = sum(np.abs(gen[i].astype(np.float32) - real) for i in range(m)) / m
    for i in range(m):
        for j in range(i + 1, m):
            ens -= np.abs(gen[i].astype(np.float32) - gen[j]) / (m * (m - 1))
    active = mask > 0
    crps = float(np.sum(ens[:, :, active] * w[None, :, None, None]) / (ens[:, :, active].size / w.size * w.sum()))
    f_gen = np.asarray(idct(member, axis=2))                 # time domain fast part [W,N,K,C]
    f_real = np.asarray(idct(real, axis=2))
    out = {"crps": crps, "variables": {}}
    groups = {v: np.array([n.split(":")[0] == v for n in names]) for v in dict.fromkeys(n.split(":")[0] for n in names)}

    def grid(x):
        return x.reshape(x.shape[0], ny, nx, *x.shape[2:])   # [W,Y,X,K,c]

    wl = w.reshape(ny, nx)[None, :, :, None, None]

    def corr(a, b, ww):
        a = a - a.mean(); b = b - b.mean()
        return float(np.sum(ww * a * b) / np.sqrt(np.sum(ww * a * a) * np.sum(ww * b * b)))

    def stats(f):
        g = grid(f)
        result = {}
        for var, sel in groups.items():
            x = g[..., sel]
            wx = np.broadcast_to(wl, x.shape)
            power = np.abs(np.fft.rfft(x, axis=2)) ** 2
            bands = [float(power[:, :, a:b].mean()) for a, b in ((1, 5), (5, 17), (17, 65))]
            xc = x - x.mean()
            result[var] = {
                "zonal": corr(x, np.roll(x, -1, 2), wx),
                "meridional": corr(x[:, :-1], x[:, 1:], wx[:, :-1]),
                "vertical": corr(x[..., :-1], x[..., 1:], wx[..., :-1]),
                "lag1_time": corr(x[:, :, :, :-1], x[:, :, :, 1:], wx[:, :, :, :-1]),
                "kurtosis": float(np.mean(xc ** 4) / np.mean(xc ** 2) ** 2 - 3),
                "bands": bands}
        t, q = g[..., groups["temperature"]], g[..., groups["specific_humidity"]]
        result["tq_corr"] = corr(t, q, np.broadcast_to(wl, t.shape))
        return result
    sg, sr = stats(f_gen), stats(f_real)
    out["tq_corr"] = {"generated": sg.pop("tq_corr"), "real": sr.pop("tq_corr")}
    for var in groups:
        a, b = sg[var], sr[var]
        out["variables"][var] = {
            "neighbour_corr": {"zonal": [a["zonal"], b["zonal"]], "meridional": [a["meridional"], b["meridional"]]},
            "vertical_corr": [a["vertical"], b["vertical"]], "lag1_time": [a["lag1_time"], b["lag1_time"]],
            "kurtosis": [a["kurtosis"], b["kurtosis"]],
            "zonal_power_ratio": [x / y for x, y in zip(a["bands"], b["bands"])]}
    return out


def multivariate_scores(gen, real, weights, names, k, c, p=0.5):
    """Variogram score over neighbour pairs and fair energy score, per variable.

    gen [S,W,N,K,C] (float16), real [W,N,K,C] coefficients; scored on the
    time-domain fast part. Lower is better for both.
    """
    ny, nx = weights["grid"]
    wl = weights["w"].reshape(1, ny, nx, 1, 1)
    to_field = lambda x: np.asarray(idct(np.asarray(x, np.float32), axis=2)).reshape(x.shape[0], ny, nx, k, c)
    y = to_field(real)
    members = [to_field(g).astype(np.float16) for g in gen]
    groups = {v: np.flatnonzero([n.split(":")[0] == v for n in names]) for v in dict.fromkeys(n.split(":")[0] for n in names)}
    pairs = {"zonal": lambda f: (f, np.roll(f, -1, 2)), "meridional": lambda f: (f[:, :-1], f[:, 1:]),
             "vertical": lambda f: (f[..., :-1], f[..., 1:]), "time_6h": lambda f: (f[:, :, :, :-1], f[:, :, :, 1:])}
    out = {}
    for var, idx in groups.items():
        yv = y[..., idx]
        mv = [np.asarray(f[..., idx], np.float32) for f in members]
        vs = {}
        for name, split in pairs.items():
            a, b = split(yv)
            ww = np.broadcast_to(wl, yv.shape)
            wa, _ = split(ww)
            observed = np.abs(a - b) ** p
            expected = sum(np.abs(np.subtract(*split(f))) ** p for f in mv) / len(mv)
            vs[name] = float(np.sum(wa * (observed - expected) ** 2) / np.sum(wa))
        sw = np.sqrt(np.broadcast_to(wl, yv.shape))
        norm = lambda d: np.sqrt(np.sum((sw * d) ** 2, axis=(1, 2, 3, 4)) / d[0].size)   # per window
        m = len(mv)
        es = sum(norm(f - yv) for f in mv) / m
        es = es - sum(norm(mv[i] - mv[j]) for i in range(m) for j in range(m) if i != j) / (2 * m * (m - 1))
        out[var] = {"variogram": vs, "energy": float(np.mean(es))}
    return out


def fit_asinh_scale(coeff, mask, k, c, windows=12):
    """Per-channel s with zero excess kurtosis of s*asinh(x/s) on training data."""
    x = np.concatenate([np.asarray(coeff[i], np.float32).reshape(-1, k, c)[:, mask > 0]
                        for i in range(0, len(coeff), max(1, len(coeff) // windows))])   # [n, K', C]
    x = x.reshape(-1, c)
    std = x.std(axis=0)
    factors = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 1e3])
    best = np.empty(c)
    for j in range(c):
        kurt = []
        for f in factors:
            s_ = f * std[j]
            yv = s_ * np.arcsinh(x[:, j] / s_)
            yv = yv - yv.mean()
            kurt.append(abs(np.mean(yv**4) / np.mean(yv**2)**2 - 3))
        best[j] = factors[int(np.argmin(kurt))] * std[j]
    return np.tile(best[None], (k, 1)).reshape(-1).astype(np.float32)   # [K*C]


class Mapped:
    """Array view that applies ``fn`` to every item read."""
    def __init__(self, array, fn):
        self.array, self.fn, self.shape = array, fn, array.shape

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.fn(np.asarray(self.array[index], np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=["current", "mlp", "mlp_corr", "unet", "unet_attn", "gaussian"])
    parser.add_argument("--corr-km", type=float, default=None)
    parser.add_argument("--x0-clip", type=float, default=X0_CLIP)
    parser.add_argument("--transform", choices=["none", "asinh"], default="none")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--train-stride", type=int, default=1, help="use every n-th training window")
    args = parser.parse_args()
    archive = ResidualArchive(args.archive)
    checkpoint = load_checkpoint(args.checkpoint)
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    data, mask, (ny, nx, k, c) = build_dataset(archive, checkpoint, work, args.train_stride)
    m = jnp.asarray(np.tile(mask[:, None], (1, c)).reshape(1, 1, k * c), jnp.float32)
    flat = lambda d: (d[0], d[1].reshape(d[1].shape[0], ny * nx, k * c))
    flat_data = {s: flat(d) for s, d in data.items()}
    vcond, vcoeff = flat_data["validation"]
    real = np.asarray(vcoeff, np.float32)                        # [W,N,KC]
    forward = inverse = lambda x: x
    if args.transform == "asinh":
        scale = fit_asinh_scale(flat_data["train"][1], mask, k, c)
        forward = lambda x: scale * np.arcsinh(x / scale)
        inverse = lambda x: scale * np.sinh(np.asarray(x, np.float32) / scale)
        flat_data = {split: (cond_, Mapped(coeff_, forward)) for split, (cond_, coeff_) in flat_data.items()}
        vcond, vcoeff = flat_data["validation"]
    lat = np.deg2rad(archive.latitudes)
    w = np.repeat(np.cos(lat), nx); w = w / w.mean()
    weights = {"w": w, "grid": (ny, nx)}
    names = list(archive.channel_names)
    from dinosaur import spherical_harmonic
    from neuralgcm_mz.spatial import spherical_noise
    length = args.corr_km if args.corr_km is not None else float(
        (checkpoint.get("noise_model") or {}).get("correlation_km_fast", 300.))
    correlated = spherical_noise(spherical_harmonic.Grid.TL63(), length)
    white = lambda key, shape: jax.random.normal(key, shape)

    def corr_draw(key, shape):
        # [B,N,F] unit fields, correlated across columns, independent across B*F.
        fields = correlated(key, shape[0] * shape[2])            # [B*F, Y, X]
        return jnp.asarray(fields).reshape(shape[0], shape[2], ny * nx).transpose(0, 2, 1)

    train_dims = (vcond.shape[-1], k * c)
    models = {"mlp": (mlp, lambda key: init_mlp(key, sum(train_dims), k * c), 1e-3),
              "unet": (lambda p, x, cc, t: unet(p, x, cc, t, (ny, nx)),
                       lambda key: init_unet(key, sum(train_dims), k * c), 5e-4),
              "unet_attn": (lambda p, x, cc, t: unet(p, x, cc, t, (ny, nx)),
                            lambda key: init_unet(key, sum(train_dims), k * c, attention=True), 5e-4)}
    n_var = len(dict.fromkeys(n.split(":")[0] for n in names))
    n_lev = c // n_var
    models["unet3d"] = (lambda p, x, cc, t: unet3d(p, x, cc, t, (ny, nx), k, n_var, n_lev),
                        lambda key: init_unet3d(key, k * n_var + 2 * n_var + 2, k * n_var), 5e-4)
    # Gaussian skip: eps_hat = base + network, where base is the exact denoiser
    # for independent Gaussian coefficients with the training variance. With a
    # zero network the sampler is that Gaussian; the network only learns the
    # departures (tails, spatial and cross-channel structure, conditioning).
    # Without it a 128- or 96-wide bottleneck cannot even pass x_t through.
    tc = flat_data["train"][1]
    sigma2 = jnp.asarray(np.mean([np.mean(np.square(tc[i], dtype=np.float32), 0)
                                  for i in range(0, len(tc), 4)], 0), jnp.float32)   # [KC]
    abar_all = schedule(STEPS)[2]

    def with_skip(model):
        def skipped(p, x, cc, t):
            a = abar_all[jnp.round(t * (STEPS - 1)).astype(jnp.int32)][:, None, None]
            base = jnp.sqrt(1 - a) / (a * sigma2 + 1 - a) * x
            return base + (0 if p is None else model(p, x, cc, t))
        return skipped
    for name in list(models):
        model, init, lr = models[name]
        models[name + "_skip"] = (with_skip(model), init, lr)
    report = {"archive": args.archive, "checkpoint": args.checkpoint, "train_windows": len(flat_data["train"][1]),
              "validation_windows": len(real), "steps": args.steps, "members": args.members,
              "fast_correlation_km": length, "x0_clip": args.x0_clip, "transform": args.transform,
              "variants": {}}
    trained = {}
    for name in [n for n in models]:
        if name not in args.variants and name + "_corr" not in args.variants:
            continue
        model, init, lr = models[name]
        params = init(jax.random.PRNGKey(7))
        count = int(sum(np.size(x) for x in jax.tree_util.tree_leaves(params)))
        tag = "" if args.transform == "none" else f"_{args.transform}"
        saved = work / f"params_{name}{tag}_{args.steps}.pkl"  # our own scratch file
        if saved.exists():
            with open(saved, "rb") as stream:
                params, best, trace = pickle.load(stream)
        else:
            params, best, trace = train(name, model, params, flat_data, m, args.steps, lr, args.batch, 11)
            with open(saved, "wb") as stream:
                pickle.dump((jax.device_get(params), best, trace), stream)
        trained[name] = (model, params)
        report["variants"][name] = {"parameters": count, "eps_mse": best, "trace": trace}
        if name + "_corr" in args.variants:
            report["variants"][name + "_corr"] = {"parameters": count, "eps_mse": best}
    if "base" in args.variants:
        evaluate = jax.jit(lambda key, x0, cc: loss_fn(with_skip(None), None, key, x0, cc, m))
        vkeys = jax.random.split(jax.random.PRNGKey(12), len(vcoeff))  # same keys as train() validation
        report["variants"]["base"] = {"parameters": 0, "eps_mse": float(np.mean([
            float(evaluate(vkeys[i], jnp.asarray(vcoeff[i:i+1], jnp.float32), jnp.asarray(vcond[i:i+1], jnp.float32)))
            for i in range(len(vcoeff))]))}

    full_history = np.load(work / "validation_history.npy")
    cp = checkpoint["diffusion_params"]
    current = lambda p, x, cc, t: jax.vmap(lambda xx, hh, tt: denoise(p, xx.reshape(-1, k, c), hh, jnp.full((xx.shape[0],), tt)
                                                                       ).reshape(xx.shape))(x, cc, t)
    gain_half = np.arange(len(real)) % 2 == 0
    for name in args.variants:
        start = time.time()
        if name == "gaussian":
            std = np.sqrt(np.mean(np.square(real[gain_half]), axis=(0, 1)))
            rng = np.random.default_rng(5)
            gen = np.zeros((args.members,) + real.shape, np.float16)
            for s_ in range(args.members):
                gen[s_] = rng.standard_normal(real.shape, np.float32) * std
        else:
            if name == "base":   # Gaussian-skip denoiser with no network
                model, params, draw, conds = with_skip(None), None, white, vcond
            elif name == "current":
                model, params, draw = current, cp, corr_draw
                conds = full_history
            else:
                model, params = trained[name.removesuffix("_corr")]
                draw = corr_draw if name.endswith("_corr") else white
                conds = vcond
            run = jax.jit(lambda key, cc: sample(model, params, key, cc, m, (1, ny * nx, k * c), draw, args.x0_clip))
            gen = np.zeros((args.members,) + real.shape, np.float16)  # [S,W,N,KC]
            for i in range(len(real)):
                cc = jnp.asarray(conds[i:i+1], jnp.float32)
                for s_ in range(args.members):
                    gen[s_, i] = inverse(np.asarray(run(jax.random.PRNGKey(1000 * s_ + i), cc))[0])
        # Amplitude gain per (mode, channel) on half the windows, as in training.
        sg = np.sqrt(np.mean(np.square(gen[:, gain_half], dtype=np.float32), axis=(0, 1, 2)))
        sr = np.sqrt(np.mean(real[gain_half] ** 2, axis=(0, 1)))
        gain = np.where(sg > 0, sr / np.maximum(sg, 1e-12), 1.0)
        gen = (gen * gain.astype(np.float16)).astype(np.float16)
        held = ~gain_half
        shape5 = lambda x: x.reshape(x.shape[:-1] + (k, c))
        result = scores(shape5(gen[:, held]), shape5(real[held]), weights, names, mask)
        multi = multivariate_scores(shape5(gen[:, held]), shape5(real[held]), weights, names, k, c)
        for var, value in multi.items():
            result["variables"][var].update(value)
        result["raw_gain_median"] = float(np.median(gain[gain != 1.0])) if np.any(gain != 1.0) else 1.0
        result["sample_seconds"] = time.time() - start
        report["variants"].setdefault(name, {}).update(result)
        print(name, json.dumps({kk: v for kk, v in result.items() if kk != "variables"}), flush=True)
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
