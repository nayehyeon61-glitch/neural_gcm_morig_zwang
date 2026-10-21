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


def init_denoiser(key, history_steps, channels, block_steps, hidden=128):
    a, b, c = jax.random.split(key, 3)
    inputs = (2 * history_steps + block_steps) * channels + 4
    return {
        "w1": jax.random.normal(a, (inputs, hidden)) / jnp.sqrt(float(inputs)),
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(b, (hidden, hidden)) / jnp.sqrt(float(hidden)),
        "b2": jnp.zeros(hidden),
        "w3": jax.random.normal(c, (hidden, block_steps, channels)) * 0.01,
        "b3": jnp.zeros((block_steps, channels)),
    }


def denoise(params, noisy, history, time):
    t = time[:, None]
    embedding = jnp.concatenate((t, t*t, jnp.sin(jnp.pi*t), jnp.cos(jnp.pi*t)), -1)
    x = jnp.concatenate((noisy.reshape((noisy.shape[0], -1)),
                         history.reshape((history.shape[0], -1)), embedding), -1)
    x = jax.nn.silu(x @ params["w1"] + params["b1"])
    x = jax.nn.silu(x @ params["w2"] + params["b2"])
    return jnp.einsum("bh,hkc->bkc", x, params["w3"]) + params["b3"]


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


def sample(params, key, history, mask, steps=32):
    """One sample per physical block; never call inside ODE RHS/stages."""
    beta, alpha, abar = schedule(steps)
    k, c = params["b3"].shape
    shape = (history.shape[0], k, c)
    key, initial_key = jax.random.split(key)
    m = mask[None, :, None]
    initial = jax.random.normal(initial_key, shape) * m

    def reverse(i, carry):
        x, rng = carry
        t = steps - 1 - i
        rng, noise_key = jax.random.split(rng)
        time = jnp.full((shape[0],), t / (steps - 1))
        eps = denoise(params, x, history, time) * m
        mean = (x - beta[t] / jnp.sqrt(1-abar[t]) * eps) / jnp.sqrt(alpha[t])
        previous = jnp.where(t > 0, abar[jnp.maximum(t-1, 0)], 1.)
        variance = beta[t] * (1-previous) / (1-abar[t])
        noise = jax.random.normal(noise_key, shape)
        x = (mean + jnp.where(t > 0, jnp.sqrt(variance) * noise, 0)) * m
        return x, rng

    return jax.lax.fori_loop(0, steps, reverse, (initial, key))[0]
