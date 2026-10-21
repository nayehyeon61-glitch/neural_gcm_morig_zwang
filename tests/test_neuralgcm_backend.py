"""Internal tendency integration against the actual upstream mini checkpoint.

Run explicitly with RUN_NEURALGCM_INTEGRATION=1. This test compiles the real
spectral dynamical core on CPU and is heavier than closure unit tests.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NEURALGCM_INTEGRATION") != "1",
    reason="Set RUN_NEURALGCM_INTEGRATION=1 to compile real NeuralGCM mini checkpoint",
)


def test_real_neuralgcm_internal_tendency_and_zero_identity():
    jax = pytest.importorskip("jax")
    neuralgcm = pytest.importorskip("neuralgcm")
    from neuralgcm import demo
    from neuralgcm_mz.neuralgcm_backend import NeuralGCMBackend

    model = neuralgcm.PressureLevelModel.from_checkpoint(demo.load_checkpoint_tl63_stochastic())
    backend = NeuralGCMBackend(model)
    data = demo.load_data(model.data_coords).isel(time=0)
    inputs = model.inputs_from_xarray(data)
    forcing = model.forcings_from_xarray(data)
    state = backend.encode(inputs, forcing, rng_key=jax.random.PRNGKey(7))
    fields = backend.sigma_fields(state)
    assert fields["temperature"].shape == backend.sigma_shape
    assert np.all(np.isfinite(fields["temperature"]))
    assert backend.timestep_seconds > 0

    baseline = backend.step(state, forcing)
    zero = backend.step(state, forcing, {"temperature": np.zeros(backend.sigma_shape, np.float32)})
    jax.block_until_ready(zero)
    for a, b in zip(jax.tree_util.tree_leaves(baseline), jax.tree_util.tree_leaves(zero)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    heated = backend.step(state, forcing, {"temperature": np.full(backend.sigma_shape, 1e-4, np.float32)})
    heat = backend.sigma_fields(heated)["temperature"]
    base = backend.sigma_fields(baseline)["temperature"]
    assert np.isfinite(heat).all()
    # A real internal K/s tendency must warm the state before decoding.
    assert float(np.mean(heat - base)) > 0.01
    with pytest.raises(ValueError, match="sigma nodal shape"):
        backend.step(state, forcing, {"temperature": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="Unknown SI tendency"):
        backend.step(state, forcing, {"geopotential": np.zeros(backend.sigma_shape)})
    with pytest.raises(ValueError, match="positive multiple"):
        backend.advance_interval(state, forcing, backend.timestep_seconds / 2)
