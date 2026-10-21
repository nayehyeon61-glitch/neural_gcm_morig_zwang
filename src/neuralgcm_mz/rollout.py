"""Causal block-noise controller; independent memory per ensemble member."""
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np

from .closure import predict_closure
from .diffusion import sample
from .spectral import highpass_mask, idct
from .training import normalize_history, denormalize_tendency
from .resources import check_process_memory, memory_preflight


class ClosureController:
    def __init__(self, checkpoint, states, past_corrections, *, seed=0,
                 correction_scale=0.1, tendency_clip_std=5., tile_columns=256,
                 mode="full", memory_limit_gb=96):
        """Histories [H,C,Y,X], ending at issue time; residuals end just before it.

        correction_scale is an experimental, validation-selected parameter.
        A value of zero exactly preserves the base trajectory. A single gain
        per column/channel bounds the whole time block, preserving the fast
        component's spectral support. This is not a conservation law.
        """
        if not 0 <= correction_scale <= 1 or not np.isfinite(tendency_clip_std) or tendency_clip_std <= 0 or tile_columns < 1:
            raise ValueError("Invalid correction strength, tendency bound, or tile size")
        if mode not in {"full", "memory", "off"}:
            raise ValueError("mode must be full, memory, or off")
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
        self.key = jax.random.PRNGKey(seed)
        self.scale = correction_scale
        self.clip = tendency_clip_std
        self.tile = tile_columns
        self.mode = mode
        self.memory_gib = memory_limit_gb * 1e9 / 1024**3
        c,y,x=self.shape
        memory_preflight(channels=c,latitude=y,longitude=x,history_steps=self.h,
            future_steps=self.config['block_steps'],model_bytes=check_process_memory(self.memory_gib),
            budget_gib=self.memory_gib)
        self.block = None
        self.position = 0
        self.steps = 0
        self.block_number = 0
        self.mask = highpass_mask(self.config["block_steps"], self.config["dt_hours"],
                                  self.config["max_period_hours"])
        self._draw = jax.jit(lambda p, key, history: sample(
            p, key, history, self.mask, self.config["diffusion_steps"]))
        self._mean = jax.jit(predict_closure)

    def _new_block(self):
        check_process_memory(self.memory_gib)
        c, y, x = self.shape
        states = np.stack(self.states).transpose(2,3,0,1).reshape(y*x,self.h,c)
        past = np.stack(self.corrections).transpose(2,3,0,1).reshape(y*x,self.h,c)
        context = normalize_history(states, past, self.checkpoint["normalizer"])
        k = self.config["block_steps"]
        block = np.empty((y*x,k,c), np.float32)
        block_key = jax.random.fold_in(self.key, self.block_number)
        for left in range(0,y*x,self.tile):
            right = min(left+self.tile,y*x)
            h = jnp.asarray(context[left:right])
            mean = self._mean(self.checkpoint["closure_params"],h)
            if self.mode == "full":
                coeff = self._draw(self.checkpoint["diffusion_params"],
                                   jax.random.fold_in(block_key,left),h)
                mean = mean + idct(coeff,axis=1)
            values = np.asarray(mean)
            gain = np.minimum(1., self.clip / np.maximum(np.max(np.abs(values),axis=1,keepdims=True),1e-12))
            bounded = values * gain
            block[left:right] = self.scale * denormalize_tendency(bounded,self.checkpoint["normalizer"])
        if not np.isfinite(block).all():
            raise FloatingPointError("Closure generated nonfinite tendencies")
        self.block = block.reshape(y,x,k,c).transpose(2,3,0,1)
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
        """Update from the forecast itself, never a future observed residual."""
        if next_state.shape != self.shape or applied_tendency.shape != self.shape:
            raise ValueError("Rollout history field shape mismatch")
        if not np.isfinite(next_state).all():
            raise FloatingPointError("Forecast became nonfinite")
        self.states.append(np.asarray(next_state, np.float32))
        self.corrections.append(np.asarray(applied_tendency, np.float32))
        if self.steps > 0 and self.block is not None:
            self.position += 1
        self.steps += 1


def pack_sigma(fields, variables):
    """[sigma,lon,lat] dictionary -> flattened [variable*sigma,lat,lon]."""
    return np.concatenate([np.asarray(fields[v]).transpose(0,2,1) for v in variables],axis=0).astype(np.float32)


def unpack_sigma(values, variables, levels):
    if values.shape[0] != len(variables)*levels:
        raise ValueError("Flattened sigma channel count mismatch")
    return {v: values[i*levels:(i+1)*levels].transpose(0,2,1) for i,v in enumerate(variables)}
