"""NeuralGCM 1.1.2 bridge: SI sigma-grid forcing inside learned physics.

The hook uses Haiku's public ``intercept_methods`` API. It intercepts the
upstream DycoreWithPhysicsCorrector, adds the closure tendency to the learned
physics tendency, then calls the unchanged upstream IMEX integrator. It never
adds noise to a decoded forecast. Samples are held fixed over solver stages.

The adapter intentionally supports the checkpoint-based PressureLevelModel API
and DycoreWithPhysicsCorrector; unsupported integrators fail rather than silently
falling back to a post-processing correction. NeuralGCM and its checkpoints keep
their original licenses. Checkpoints are pickle: load trusted files only.
"""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


def _upstream():
    """Lazy dependency import, compatible with the named legacy API only."""
    import haiku as hk
    import neuralgcm
    from dinosaur import pytree_utils, spherical_harmonic, xarray_utils
    try:
        from neuralgcm.legacy import api, correctors, gin_utils
    except ImportError:
        from neuralgcm import api, correctors, gin_utils
    return hk, neuralgcm, pytree_utils, spherical_harmonic, xarray_utils, api, correctors, gin_utils


class NeuralGCMBackend:
    """A frozen base model with differentiable external physics tendencies.

    ``tendencies_si`` arrays have shape [sigma, longitude, latitude] on the
    *model* grid, not ERA5 pressure levels. Temperature is K/s, wind is m/s²,
    specific humidity and cloud tracers are kg/kg/s. Missing channels are zero.
    The base model's own memory/randomness are retained. Closure memory belongs
    to the caller and must persist between calls, independently per ensemble.
    """

    def __init__(self, model: Any):
        self.model = model
        (self.hk, _, self.pytree_utils, self.spherical_harmonic,
         self.xarray_utils, self.api, self.correctors, self.gin_utils) = _upstream()
        if "DycoreWithPhysicsCorrector" not in model.gin_config:
            raise ValueError("Checkpoint must use DycoreWithPhysicsCorrector for internal forcing")
        self._step_compiled = jax.jit(self._step_impl)
        self._modal_compiled = jax.jit(self._modal_correction)
        self._fields_compiled = jax.jit(self._sigma_fields)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path | Mapping[str, Any]):
        """Load a trusted upstream checkpoint or an already loaded dict."""
        if isinstance(checkpoint, (str, Path)):
            with Path(checkpoint).open("rb") as handle:
                checkpoint = pickle.load(handle)
        *_, api, _, _ = _upstream()
        return cls(api.PressureLevelModel.from_checkpoint(checkpoint))

    @property
    def timestep_seconds(self) -> float:
        return float(self.model.timestep / np.timedelta64(1, "s"))

    @property
    def sigma_shape(self) -> tuple[int, int, int]:
        coords = self.model.model_coords
        return (coords.vertical.layers, *coords.horizontal.nodal_shape)

    def encode(self, inputs, forcings, rng_key=None):
        return self.model.encode(inputs, forcings, rng_key=rng_key)

    def decode(self, state, forcings):
        return self.model.decode(state, forcings)

    def sigma_fields(self, state) -> dict[str, jax.Array]:
        """Return physical SI fields for causal conditioning/encoded residuals."""
        return self._fields_compiled(state)

    def _sigma_fields(self, state):
        s = state.state if hasattr(state, "state") else state
        fields, _ = self.pytree_utils.as_dict(s)
        grid = self.model.model_coords.horizontal
        cu, cv = self.spherical_harmonic.get_cos_lat_vector(
            fields["vorticity"], fields["divergence"], grid)
        ref = self.model._structure.specs.aux_features[self.xarray_utils.REF_TEMP_KEY]
        t = grid.to_nodal(fields["temperature_variation"]) + jnp.asarray(ref)[:, None, None]
        output = {
            "temperature": self.model.from_nondim_units(t, "K"),
            "u": self.model.from_nondim_units(grid.to_nodal(cu) / grid.cos_lat, "m/s"),
            "v": self.model.from_nondim_units(grid.to_nodal(cv) / grid.cos_lat, "m/s"),
        }
        for key, value in fields["tracers"].items():
            output[key] = grid.to_nodal(value)
        return output

    def _modal_correction(self, state, tendencies_si):
        s = state.state if hasattr(state, "state") else state
        fields, restore = self.pytree_utils.as_dict(s)
        result = jax.tree_util.tree_map(jnp.zeros_like, fields)
        grid = self.model.model_coords.horizontal
        zero = jnp.zeros(self.sigma_shape, dtype=jnp.float32)
        t = tendencies_si.get("temperature", zero)
        result["temperature_variation"] = grid.to_modal(self.model.to_nondim_units(t, "K/s"))
        # Exact spherical velocity->divergence/vorticity mapping used by upstream
        # ToModalWithDivCurlTransform, including its cos(latitude) convention.
        u = self.model.to_nondim_units(tendencies_si.get("u", zero), "m/s^2")
        v = self.model.to_nondim_units(tendencies_si.get("v", zero), "m/s^2")
        uv = (grid.to_modal(u / grid.cos_lat), grid.to_modal(v / grid.cos_lat))
        result["divergence"] = grid.div_cos_lat(uv)
        result["vorticity"] = grid.curl_cos_lat(uv)
        for key in fields["tracers"]:
            result["tracers"][key] = grid.to_modal(
                self.model.to_nondim_units(tendencies_si.get(key, zero), "1/s"))
        return restore(result)

    def _step_impl(self, params, state, forcings, correction):
        seen = []

        def interceptor(next_f, args, kwargs, context):
            if (isinstance(context.module, self.correctors.DycoreWithPhysicsCorrector)
                    and context.method_name == "__call__"):
                seen.append(True)
                # Haiku intercept arguments exclude self. The upstream step
                # passes (state, tendencies, forcing) positionally.
                args = list(args)
                base = args[1] if len(args) > 1 else kwargs["tendencies"]
                # Some checkpoints wrap the dycore in CustomCoordsCorrector
                # (e.g. padded vs unpadded spectral grids). Use its *actual*
                # coords, not guessed shape slicing, before adding tendencies.
                from dinosaur import coordinate_systems
                interpolate = coordinate_systems.get_spectral_interpolate_fn(
                    self.model.model_coords, context.module.coords)
                local_correction = interpolate(correction)
                add = lambda a, b: b if a is None else a + b
                combined = jax.tree_util.tree_map(add, base, local_correction, is_leaf=lambda x: x is None)
                if len(args) > 1:
                    args[1] = combined
                else:
                    kwargs = {**kwargs, "tendencies": combined}
                return next_f(*args, **kwargs)
            return next_f(*args, **kwargs)

        with self.gin_utils.specific_config(self.model.gin_config):
            sim_time = self.api._sim_time_from_state(state)
            forcing_data = self.model._squeeze_level_from_forcings(forcings)
            forcing_data = self.api._prepend_dummy_time_axis(forcing_data)
            forcing = self.model._structure.forcing_fn(params, None, forcing_data, sim_time)
            with self.hk.intercept_methods(interceptor):
                result = self.model._structure.advance_fn(params, None, state, forcing)
        if not seen:
            raise RuntimeError("No DycoreWithPhysicsCorrector was called; closure was not injected")
        return result

    def step(self, state, forcings, tendencies_si: Mapping[str, Any] | None = None):
        """Advance one native timestep, injecting SI tendencies inside IMEX.

        Use ``None`` for the original unmodified baseline. All supplied values
        must already be on sigma levels; passing pressure-level residuals here
        is a coordinate error, even if the array dimensions happen to match.
        """
        if tendencies_si is None:
            return self.model.advance(state, forcings)
        s = state.state if hasattr(state, "state") else state
        fields, _ = self.pytree_utils.as_dict(s)
        allowed = {"temperature", "u", "v", *fields["tracers"]}
        unknown = set(tendencies_si) - allowed
        if unknown:
            raise ValueError(f"Unknown SI tendency fields: {sorted(unknown)}")
        for key, value in tendencies_si.items():
            if np.shape(value) != self.sigma_shape:
                raise ValueError(f"{key}: expected sigma nodal shape {self.sigma_shape}, got {np.shape(value)}")
        modal = self._modal_compiled(state, tendencies_si)
        return self._step_compiled(self.model.params, state, forcings, modal)

    def persisted_forcings(self, state, forcings):
        """Reuse supplied boundary values at the current physical time.

        The timestamp labels the time at which the persistence assumption is
        evaluated. It is not a newly observed SST/ice field. Upstream dynamic
        forcing rejects stale timestamps, so changing only these labels is
        necessary when intentionally prescribing persistent boundaries.
        """
        return {**forcings, "sim_time": self.api._sim_time_from_state(state)}

    def advance_interval(self, state, forcings, seconds, tendencies_si=None):
        """Integrate with a fixed sample and persisted boundary values.

        Each native step relabels the same boundary field at its current time;
        no future SST or sea-ice observation is consulted.
        """
        steps = float(seconds) / self.timestep_seconds
        if seconds <= 0 or not np.isclose(steps, round(steps), rtol=0, atol=1e-7):
            raise ValueError("Interval must be a positive multiple of the native timestep")
        # A Python streaming loop avoids storing the full rollout trajectory.
        for _ in range(round(steps)):
            state = self.step(state, self.persisted_forcings(state, forcings), tendencies_si)
        return state
