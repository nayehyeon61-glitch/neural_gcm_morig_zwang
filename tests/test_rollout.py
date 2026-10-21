import numpy as np
import jax
from neuralgcm_mz.closure import init_closure
from neuralgcm_mz.diffusion import init_denoiser
from neuralgcm_mz.rollout import ClosureController, pack_sigma, unpack_sigma


def checkpoint():
    return dict(config=dict(history_steps=3,block_steps=8,dt_hours=6,max_period_hours=48,diffusion_steps=8),
        channel_names=['t','u'], closure_params=init_closure(jax.random.PRNGKey(0),3,2,8,8),
        diffusion_params=init_denoiser(jax.random.PRNGKey(1),3,2,8,8),
        normalizer=dict(state_mean=np.zeros(2),state_std=np.ones(2),residual_mean=np.zeros(2),residual_std=np.ones(2)))


def test_first_interval_base_reproducibility_and_40_day_history():
    shape=(3,2,2,3)
    a=ClosureController(checkpoint(),np.zeros(shape),np.zeros(shape),seed=9)
    b=ClosureController(checkpoint(),np.zeros(shape),np.zeros(shape),seed=9)
    np.testing.assert_array_equal(a.tendency(),0)
    state=np.zeros(shape[1:],np.float32)
    for _ in range(160):
        x=a.tendency(); y=b.tendency()
        np.testing.assert_array_equal(x,y)
        state=state+0.01*x
        a.observe_generated(state,x); b.observe_generated(state,y)
    assert len(a.states)==3 and len(a.corrections)==3 and a.steps==160
    assert np.isfinite(state).all()


def test_sigma_layout_roundtrip_and_off_mode():
    fields={'temperature':np.arange(24).reshape(2,4,3),'u':np.ones((2,4,3))}
    packed=pack_sigma(fields,['temperature','u'])
    restored=unpack_sigma(packed,['temperature','u'],2)
    for k in fields: np.testing.assert_array_equal(restored[k],fields[k])
    c=ClosureController(checkpoint(),np.zeros((3,2,2,3)),np.zeros((3,2,2,3)),mode='off')
    for _ in range(3):
        t=c.tendency(); np.testing.assert_array_equal(t,0); c.observe_generated(t,t)
