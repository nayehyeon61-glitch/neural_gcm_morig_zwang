"""Causal block-noise controller; independent memory per ensemble member."""
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np

from .closure import predict_closure
from .diffusion import reverse_step
from .spectral import highpass_mask, idct
from .training import normalize_history, denormalize_tendency
from .resources import check_process_memory, memory_preflight
from .spatial import independent_noise


class ClosureController:
    def __init__(self, checkpoint, states, past_corrections, *, seed=0,
                 correction_scale=0.1, tendency_clip_std=5., tile_columns=256,
                 mode="full", memory_limit_gb=96, noise_fn=None,
                 slow_noise=True, slow_noise_scale=1.0):
        """Histories [H,C,Y,X], ending at issue time; residuals end just before it.

        correction_scale is an experimental, validation-selected parameter.
        A value of zero exactly preserves the base trajectory. A single gain
        per column/channel bounds the whole time block, preserving the fast
        component's spectral support. This is not a conservation law.

        ``noise_fn(key, n) -> [n,Y,X]`` supplies unit-variance fields for all
        stochastic draws (see ``spatial.py``); the default is spatially white.
        ``slow_noise`` enables the zero-frequency AR(1) branch when the
        checkpoint carries its fitted statistics. Modes: ``full`` (mean, fast
        and slow noise), ``memory`` (mean only), ``noise`` (noise only, the
        noise-without-memory ablation) and ``off``. ``seed`` may be an integer
        or a JAX key.
        """
        if not 0 <= correction_scale <= 1 or not np.isfinite(tendency_clip_std) or tendency_clip_std <= 0 or tile_columns < 1:
            raise ValueError("Invalid correction strength, tendency bound, or tile size")
        if mode not in {"full", "memory", "noise", "off"}:
            raise ValueError("mode must be full, memory, noise, or off")
        if not np.isfinite(slow_noise_scale) or slow_noise_scale < 0:
            raise ValueError("slow_noise_scale must be finite and nonnegative")
        self.checkpoint = checkpoint
        self.config = checkpoint["config"]
        self.h = self.config["history_steps"]
        if states.shape != past_corrections.shape or states.shape[0] != self.h:
            raise ValueError("Supply exactly H aligned state and past-correction fields")
        self.shape = states.shape[1:]
        if len(self.shape) != 3 or self.shape[0] != len(checkpoint["channel_names"]):
            raise ValueError("Closure channel count does not match checkpoint")
        self.states = deque((np.array(x, dtype=np.float32) for x in states), maxlen=self.h)
        self.corrections = deque((np.array(x, dtype=np.float32) for x in past_corrections), maxlen=self.h)
        self.key = jax.random.PRNGKey(seed) if np.ndim(seed) == 0 else jnp.asarray(seed)
        self.scale = correction_scale
        self.clip = tendency_clip_std
        self.tile = tile_columns
        self.mode = mode
        self.memory_gib = memory_limit_gb * 1e9 / 1024**3
        c,y,x=self.shape
        memory_preflight(channels=c,latitude=y,longitude=x,history_steps=self.h,
            future_steps=self.config['block_steps'],model_bytes=check_process_memory(self.memory_gib),
            budget_gib=self.memory_gib)
        self.noise_fn = noise_fn or independent_noise(y, x)
        k = self.config["block_steps"]
        noise_model = checkpoint.get("noise_model") or {}
        self.gain = np.asarray(noise_model.get("diffusion_gain", np.ones((k, c))), np.float32)
        if self.gain.shape != (k, c):
            raise ValueError("Diffusion gain does not match block/channel shape")
        self.slow_enabled = bool(slow_noise and "slow_cholesky" in noise_model and slow_noise_scale > 0)
        if self.slow_enabled:
            self.slow_cholesky = jnp.asarray(noise_model["slow_cholesky"], jnp.float32) * slow_noise_scale
            self.slow_phi = float(noise_model["slow_phi"])
            if self.slow_cholesky.shape != (c, c) or not 0 <= self.slow_phi < 1:
                raise ValueError("Invalid slow-noise statistics in checkpoint")
        self.slow_state = None
        self.block = None
        self.history_block = None
        self.position = 0
        self.steps = 0
        self.block_number = 0
        self.mask = highpass_mask(k, self.config["dt_hours"], self.config["max_period_hours"])
        steps = self.config["diffusion_steps"]
        self._reverse = jax.jit(lambda p, xt, h, t, z: reverse_step(p, xt, h, t, z, self.mask, steps))
        self._mean = jax.jit(predict_closure)

    def _fields(self, key, n):
        """[n,Y,X] unit fields -> column layout [Y*X, n]."""
        c, y, x = self.shape
        return jnp.asarray(self.noise_fn(key, n)).reshape(n, y*x).T

    def _fast_noise(self, key, context):
        """Reverse DDPM over all columns, with column-correlated Gaussian draws."""
        c, y, x = self.shape
        k = self.config["block_steps"]
        steps = self.config["diffusion_steps"]
        keys = jax.random.split(key, steps + 1)
        m = jnp.asarray(self.mask)[None, :, None]
        draw = lambda key: self._fields(key, k*c).reshape(y*x, k, c)
        coeff = draw(keys[0]) * m
        params = self.checkpoint["diffusion_params"]
        for i in range(steps):
            t = steps - 1 - i
            z = draw(keys[i + 1])
            coeff = jnp.concatenate([
                self._reverse(params, coeff[a:b], context[a:b], t, z[a:b])
                for a, b in ((a, min(a + self.tile, y*x)) for a in range(0, y*x, self.tile))])
        return idct(coeff * jnp.asarray(self.gain)[None], axis=1)

    def _slow_noise(self, key):
        """Block-constant zero-frequency innovation, AR(1) across blocks."""
        c, y, x = self.shape
        shock = self._fields(key, c) @ self.slow_cholesky.T
        if self.slow_state is None:
            self.slow_state = shock
        else:
            self.slow_state = self.slow_phi * self.slow_state + np.sqrt(1 - self.slow_phi**2) * shock
        return self.slow_state[:, None, :]

    def _new_block(self):
        check_process_memory(self.memory_gib)
        c, y, x = self.shape
        states = np.stack(self.states).transpose(2,3,0,1).reshape(y*x,self.h,c)
        past = np.stack(self.corrections).transpose(2,3,0,1).reshape(y*x,self.h,c)
        context = jnp.asarray(normalize_history(states, past, self.checkpoint["normalizer"]))
        k = self.config["block_steps"]
        fast_key, slow_key = jax.random.split(jax.random.fold_in(self.key, self.block_number))
        mean = np.zeros((y*x,k,c), np.float32)
        if self.mode in {"full", "memory"}:
            for left in range(0,y*x,self.tile):
                right = min(left+self.tile,y*x)
                mean[left:right] = np.asarray(self._mean(self.checkpoint["closure_params"],context[left:right]))
        noise = np.zeros_like(mean)
        if self.mode in {"full", "noise"}:
            noise = self._fast_noise(fast_key, context)
            if self.slow_enabled:
                noise = noise + self._slow_noise(slow_key)
            noise = np.asarray(noise)
        values = mean + noise
        gain = np.minimum(1., self.clip / np.maximum(np.max(np.abs(values),axis=1,keepdims=True),1e-12))
        norm = self.checkpoint["normalizer"]
        if self.mode == "noise":
            # Stochastic part only: no climatological bias or history mean.
            physical = values * gain * norm["residual_std"]
        else:
            physical = denormalize_tendency(values * gain, norm)
        if not np.isfinite(physical).all():
            raise FloatingPointError("Closure generated nonfinite tendencies")
        physical = physical.astype(np.float32).reshape(y,x,k,c).transpose(2,3,0,1)
        # Memory sees the closure's own residual estimate, at the scale it was
        # trained on; only the applied forcing is multiplied by the strength.
        self.history_block = physical
        self.block = self.scale * physical
        self.position = 0
        self.block_number += 1

    def tendency(self):
        """First physical interval is always pure NeuralGCM. Call once/interval."""
        if self.steps == 0 or self.mode == "off" or self.scale == 0:
            return np.zeros(self.shape, np.float32)
        if self.block is None or self.position == len(self.block):
            self._new_block()
        return self.block[self.position].copy()

    def observe_generated(self, next_state, applied_tendency):
        """Update from the forecast itself, never a future observed residual.

        The stored past correction is the unscaled closure estimate behind
        ``applied_tendency``. Intervals without a closure estimate (the pure
        first interval, ``off`` mode) store the training-mean residual, which
        normalizes to zero, rather than an out-of-distribution exact zero.
        """
        if next_state.shape != self.shape or applied_tendency.shape != self.shape:
            raise ValueError("Rollout history field shape mismatch")
        if not np.isfinite(next_state).all():
            raise FloatingPointError("Forecast became nonfinite")
        self.states.append(np.asarray(next_state, np.float32))
        if self.steps > 0 and self.block is not None:
            self.corrections.append(self.history_block[self.position].copy())
            self.position += 1
        else:
            mean = np.asarray(self.checkpoint["normalizer"]["residual_mean"], np.float32)
            self.corrections.append(np.broadcast_to(mean[:, None, None], self.shape).copy())
        self.steps += 1


def pack_sigma(fields, variables):
    """[sigma,lon,lat] dictionary -> flattened [variable*sigma,lat,lon]."""
    return np.concatenate([np.asarray(fields[v]).transpose(0,2,1) for v in variables],axis=0).astype(np.float32)


def unpack_sigma(values, variables, levels):
    if values.shape[0] != len(variables)*levels:
        raise ValueError("Flattened sigma channel count mismatch")
    return {v: values[i*levels:(i+1)*levels].transpose(0,2,1) for i,v in enumerate(variables)}
