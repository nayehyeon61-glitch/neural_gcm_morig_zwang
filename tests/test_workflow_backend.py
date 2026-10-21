"""Public API name checks and an optional real prepare/forecast smoke test."""
import os
import pickle

import numpy as np
import pytest

from neuralgcm_mz import workflow
from neuralgcm_mz.training import save_checkpoint, load_checkpoint


def test_pressure_field_names_follow_public_api():
    decoded = {
        'temperature': np.full((2, 4, 3), 280., np.float32),
        'u_component_of_wind': np.full((2, 4, 3), 12., np.float32),
        'v_component_of_wind': np.full((2, 4, 3), -3., np.float32),
        'specific_humidity': np.full((2, 4, 3), .001, np.float32),
    }
    packed = workflow._pressure_fields(decoded, workflow.VARIABLES)
    assert packed.shape == (8, 3, 4)
    np.testing.assert_array_equal(packed[2:4], 12.)
    np.testing.assert_array_equal(packed[4:6], -3.)


@pytest.mark.skipif(os.environ.get('RUN_NEURALGCM_INTEGRATION') != '1',
                    reason='Set RUN_NEURALGCM_INTEGRATION=1 for actual NeuralGCM workflow smoke')
def test_real_prepare_forecast_persistent_boundaries(tmp_path, monkeypatch):
    """Repeated demo ERA5 states test plumbing, NOT forecast skill.

    The forcing tolerance is deliberately narrowed to one hour, so two-hour
    residual preparation and a full day of inference fail if timestamps stay
    frozen while the same SST/sea-ice boundary values are persisted.
    """
    import jax
    import neuralgcm
    from neuralgcm import demo
    from neuralgcm_mz.neuralgcm_backend import NeuralGCMBackend
    from neuralgcm_mz.data import ResidualArchive

    checkpoint = demo.load_checkpoint_tl63_stochastic()
    checkpoint['model_config_str'] = checkpoint['model_config_str'].replace(
        "DynamicDataForcing.dt_tolerance = '1 year'",
        "DynamicDataForcing.dt_tolerance = '1 hour'")
    checkpoint_path = tmp_path/'mini.pkl'
    with checkpoint_path.open('wb') as handle:
        pickle.dump(checkpoint, handle)
    backend = NeuralGCMBackend.from_checkpoint(checkpoint)
    monkeypatch.setattr(workflow, '_backend', lambda config: backend)
    data = demo.load_data(backend.model.data_coords).isel(time=0)
    start = np.asarray(data.time.values).astype('datetime64[ns]')
    times = start + np.arange(4)*np.timedelta64(2, 'h')
    data = data.drop_vars('time').expand_dims(time=times)
    # Exercise native ERA5 variable/dimension spelling, not just friendly names.
    renames = {"temperature":"t", "specific_humidity":"q", "geopotential":"z",
               "u_component_of_wind":"u", "v_component_of_wind":"v",
               "sea_surface_temperature":"sst", "sea_ice_cover":"ci",
               "level":"pressure_level", "time":"valid_time"}
    data = data.rename({a:b for a,b in renames.items() if a in data})
    era5_path = tmp_path/'era5.nc'
    data.to_netcdf(era5_path)
    config = {'neuralgcm_checkpoint':str(checkpoint_path), 'era5':str(era5_path),
              'archive':str(tmp_path/'archive'), 'dt_hours':2,
              'splits':{'train':[0,1], 'validation':[1,2], 'test':[2,3]}}
    result = workflow.prepare(config)
    assert result['intervals'] == 3
    archive = ResidualArchive(config['archive'])
    assert np.isfinite(archive.states).all()
    assert np.isfinite(archive.residual).all()
    assert not (tmp_path/'archive.preparation').exists()

    # Mode=off validates the whole operational path with an explicit disabled
    # closure. No untrained synthetic weights are presented as a trained model.
    channels = len(archive.channel_names)
    closure_path = tmp_path/'disabled_closure.npz'
    save_checkpoint(closure_path, {
        'channel_names':archive.channel_names, 'dt_seconds':7200.,
        'config':{'history_steps':1,'block_steps':2,'dt_hours':2.,
                  'max_period_hours':8.,'diffusion_steps':2},
        'normalizer':{'state_mean':np.zeros(channels),'state_std':np.ones(channels),
                      'residual_mean':np.zeros(channels),'residual_std':np.ones(channels)},
        'closure_params':{},'diffusion_params':{},
        'provenance':{'archive_provenance':archive.manifest['provenance'],
                      'neuralgcm_sha256':workflow.checkpoint_sha256(checkpoint_path)},
    })
    forecast_config = {**config, 'closure_checkpoint':str(closure_path),
        'issue_time':str(times[-1]), 'days':1,'members':1,'mode':'off',
        'output':str(tmp_path/'forecast')}
    metadata = workflow.forecast(forecast_config)
    assert metadata['lead_days'] == [1]
    baseline = np.load(tmp_path/'forecast'/'baseline.npy', mmap_mode='r')
    corrected = np.load(tmp_path/'forecast'/'ensemble.npy', mmap_mode='r')
    assert np.isfinite(baseline).all()
    np.testing.assert_array_equal(baseline, corrected)
    weights = np.load(tmp_path/'forecast'/'latitude_weights.npy')
    assert weights.shape == (baseline.shape[-2],)
    assert np.all(weights > 0)

    # Untrained random parameters are used ONLY as a numerical wiring test.
    # Enable the actual history model -> DDPM -> time IDCT -> sigma unpack ->
    # internal learned-physics forcing path with a very small physical scale.
    from neuralgcm_mz.closure import init_closure
    from neuralgcm_mz.diffusion import init_denoiser
    payload = load_checkpoint(closure_path)
    payload['closure_params'] = init_closure(jax.random.PRNGKey(101), 1, channels, 2, hidden=8)
    payload['diffusion_params'] = init_denoiser(jax.random.PRNGKey(102), 1, channels, 2, hidden=8)
    sample_state = np.asarray(archive.states[0])
    payload['normalizer']['state_mean'] = sample_state.mean(axis=(1,2))
    payload['normalizer']['state_std'] = np.maximum(sample_state.std(axis=(1,2)), 1e-6)
    levels = backend.sigma_shape[0]
    payload['normalizer']['residual_std'] = np.concatenate((
        np.full(3*levels, 1e-4, np.float32), np.full(levels, 1e-8, np.float32)))
    enabled_path = tmp_path/'untrained_software_smoke_closure.npz'
    save_checkpoint(enabled_path, payload)
    enabled_config = {**forecast_config, 'mode':'full','correction_scale':1e-4,
                      'closure_checkpoint':str(enabled_path),'output':str(tmp_path/'enabled')}
    enabled_metadata = workflow.forecast(enabled_config)
    assert enabled_metadata['closure_mode'] == 'full'
    enabled_base = np.load(tmp_path/'enabled'/'baseline.npy', mmap_mode='r')
    enabled = np.load(tmp_path/'enabled'/'ensemble.npy', mmap_mode='r')
    assert np.isfinite(enabled).all()
    np.testing.assert_array_equal(enabled_base, baseline)
    assert np.max(np.abs(enabled - enabled_base)) > 0
