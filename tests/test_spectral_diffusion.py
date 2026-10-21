import numpy as np
import jax
import jax.numpy as jnp
from neuralgcm_mz import spectral, diffusion, closure


def test_dct_matches_scipy_and_preserves_energy():
    from scipy.fft import dct
    x = np.random.default_rng(1).normal(size=(3,16,4)).astype('float32')
    z = spectral.dct(x)
    np.testing.assert_allclose(z, dct(x, axis=1, norm='ortho'), atol=6e-6)
    np.testing.assert_allclose(spectral.idct(z), x, atol=6e-6)
    np.testing.assert_allclose(np.sum(z*z), np.sum(x*x), rtol=2e-6)


def test_known_period_and_zero_dc():
    matrix = spectral.dct_matrix(16)
    x = jnp.asarray(matrix[1] + 2*matrix[8])[None, :, None]
    slow, fast = spectral.split_bands(x, 6, 48)
    np.testing.assert_allclose(fast[0,:,0], 2*matrix[8], atol=5e-6)
    np.testing.assert_allclose(slow[0,:,0], matrix[1], atol=5e-6)


def test_ddpm_gradients_keys_and_band_support():
    h = jnp.ones((2, 3, 4))
    key = jax.random.PRNGKey(3)
    p = diffusion.init_denoiser(key, 3, 2, 8, 8)
    mask = spectral.highpass_mask(8, 6, 48)
    target = jnp.ones((2,8,2))
    loss, grad = jax.value_and_grad(diffusion.diffusion_loss)(p,key,target,h,mask,8)
    assert np.isfinite(loss)
    assert sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grad)) > 0
    a = diffusion.sample(p,key,h,mask,8)
    b = diffusion.sample(p,key,h,mask,8)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(np.asarray(a)[:,np.asarray(mask)==0,:],0)
    assert not np.allclose(a,diffusion.sample(p,jax.random.PRNGKey(4),h,mask,8))


def test_history_and_prior_noise_affect_closure():
    p = closure.init_closure(jax.random.PRNGKey(2), 3, 2, 8, 8)
    h = jnp.zeros((1,3,4))
    assert not np.allclose(closure.predict_closure(p,h),closure.predict_closure(p,h.at[:,0,3].set(1)))
