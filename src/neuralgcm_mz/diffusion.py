"""Conditional Gaussian DDPM over the active time-DCT subspace.

Physical residuals are not asserted Gaussian. Only q(x_s|x_0) and the prior
are Gaussian. Joint block generation supplies correlated fast trajectories.
"""
import jax
import jax.numpy as jnp


def schedule(steps):
    if steps < 2:
        raise ValueError("DDPM requires at least 2 diffusion steps")
    t = jnp.linspace(0., 1., steps + 1)
    abar = jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2
    abar = abar / abar[0]
    beta = jnp.clip(1 - abar[1:] / abar[:-1], 1e-5, 0.999)
    alpha = 1 - beta
    return beta, alpha, jnp.cumprod(alpha)


def init_denoiser(key, history_steps, channels, block_steps, hidden=128, extra=0):
    a, b, c = jax.random.split(key, 3)
    inputs = history_steps * (2 * channels + extra) + block_steps * channels + 4
    return {
        "w1": jax.random.normal(a, (inputs, hidden)) / jnp.sqrt(float(inputs)),
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(b, (hidden, hidden)) / jnp.sqrt(float(hidden)),
        "b2": jnp.zeros(hidden),
        "w3": jax.random.normal(c, (hidden, block_steps, channels)) * 0.01,
        "b3": jnp.zeros((block_steps, channels)),
    }


def with_gaussian_skip(params, coefficient_variance, steps):
    """Add the exact Gaussian denoiser as a fixed skip path.

    For independent coefficients x0 ~ N(0, s2) the optimal noise estimate is
    sqrt(1-abar) / (abar s2 + 1 - abar) * x_t. The MLP output is added on top,
    so it only learns departures from that Gaussian (conditioning, tails,
    cross-channel structure). Without the skip a 128-wide hidden layer cannot
    carry the 8 x C noisy input through: on the small real-data test the
    validation noise-prediction loss was 0.89, worse than this linear rule
    alone (about 0.38), and raw samples were five times too wide.
    """
    return {**params, "skip_variance": jnp.asarray(coefficient_variance, jnp.float32),
            "skip_abar": schedule(steps)[2]}


def denoise(params, noisy, history, time):
    t = time[:, None]
    embedding = jnp.concatenate((t, t*t, jnp.sin(jnp.pi*t), jnp.cos(jnp.pi*t)), -1)
    x = jnp.concatenate((noisy.reshape((noisy.shape[0], -1)),
                         history.reshape((history.shape[0], -1)), embedding), -1)
    x = jax.nn.silu(x @ params["w1"] + params["b1"])
    x = jax.nn.silu(x @ params["w2"] + params["b2"])
    out = jnp.einsum("bh,hkc->bkc", x, params["w3"]) + params["b3"]
    if "skip_variance" in params:
        abar = jax.lax.stop_gradient(params["skip_abar"])
        s2 = jax.lax.stop_gradient(params["skip_variance"])
        a = abar[jnp.round(time * (abar.shape[0] - 1)).astype(jnp.int32)][:, None, None]
        out = out + jnp.sqrt(1 - a) / (a * s2 + 1 - a) * noisy
    return out


def diffusion_loss(params, key, target_coeff, history, mask, steps=32):
    time_key, noise_key = jax.random.split(key)
    t = jax.random.randint(time_key, (target_coeff.shape[0],), 0, steps)
    _, _, abar = schedule(steps)
    m = mask[None, :, None]
    noise = jax.random.normal(noise_key, target_coeff.shape) * m
    a = abar[t, None, None]
    noisy = jnp.sqrt(a) * target_coeff * m + jnp.sqrt(1-a) * noise
    predicted = denoise(params, noisy, history, t / (steps-1)) * m
    # Divide by active entries, not by inactive low-frequency coefficients.
    return jnp.sum((predicted-noise)**2) / (jnp.sum(mask)*target_coeff.shape[0]*target_coeff.shape[2])


X0_CLIP = 6.0


def reverse_step(params, x, history, t, noise, mask, steps=32):
    """One ancestral DDPM step at index t, given unit Gaussian ``noise``.

    The caller owns the noise so that it may be spatially correlated across
    columns. Each column still receives marginally standard normal noise, so
    the per-column reverse process is unchanged.

    The posterior mean is formed from the implied clean sample
    x0 = (x - sqrt(1-abar) eps) / sqrt(abar), clipped to +-X0_CLIP (targets
    are unit-variance DCT coefficients). Unclipped this equals the usual
    eps form; clipped it stops an imperfect eps network from compounding
    the 1/sqrt(alpha) growth near t = T, which drove an undertrained
    sampler to hundreds of standard deviations.
    """
    beta, alpha, abar = schedule(steps)
    m = mask[None, :, None]
    time = jnp.full((x.shape[0],), t / (steps - 1))
    eps = denoise(params, x, history, time) * m
    previous = jnp.where(t > 0, abar[jnp.maximum(t-1, 0)], 1.)
    x0 = jnp.clip((x - jnp.sqrt(1-abar[t]) * eps) / jnp.sqrt(abar[t]), -X0_CLIP, X0_CLIP)
    mean = (jnp.sqrt(previous) * beta[t] * x0 + jnp.sqrt(alpha[t]) * (1-previous) * x) / (1-abar[t])
    variance = beta[t] * (1-previous) / (1-abar[t])
    return (mean + jnp.where(t > 0, jnp.sqrt(variance) * noise, 0)) * m


def sample(params, key, history, mask, steps=32):
    """One sample per physical block; never call inside ODE RHS/stages."""
    k, c = params["b3"].shape
    shape = (history.shape[0], k, c)
    key, initial_key = jax.random.split(key)
    initial = jax.random.normal(initial_key, shape) * mask[None, :, None]

    def reverse(i, carry):
        x, rng = carry
        rng, noise_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, shape)
        return reverse_step(params, x, history, steps - 1 - i, noise, mask, steps), rng

    return jax.lax.fori_loop(0, steps, reverse, (initial, key))[0]
