"""Finite-history learned closure, inspired by Hist_Bayesian_Closure.

History contains resolved state AND previous corrections. All channels are
visible to each output head. This is a learned MZ-inspired approximation;
it does not solve the exact orthogonal dynamics or enforce an FDT identity.
"""
import jax
import jax.numpy as jnp


def init_closure(key, history_steps, channels, block_steps, hidden=128):
    a, b = jax.random.split(key)
    inputs = history_steps * channels * 2
    return {
        "w1": jax.random.normal(a, (inputs, hidden)) / jnp.sqrt(float(inputs)),
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(b, (hidden, block_steps, channels)) * 0.01,
        "b2": jnp.zeros((block_steps, channels)),
    }


def predict_closure(params, history):
    """[columns, history, 2*channels] -> [columns, future_time, channels]."""
    features = history.reshape((history.shape[0], -1))
    hidden = jax.nn.silu(features @ params["w1"] + params["b1"])
    return jnp.einsum("bh,hkc->bkc", hidden, params["w2"]) + params["b2"]


def causal_context(states, corrections, state_mean, state_std,
                   residual_mean, residual_std):
    """Normalize histories without fitting any statistics at inference time."""
    return jnp.concatenate(((states - state_mean) / state_std,
                            (corrections - residual_mean) / residual_std), axis=-1)
