"""ERA5 -> native sigma residual archive -> paired 40-day NeuralGCM rollout.

Run: python -m neuralgcm_mz.workflow {prepare,forecast,verify} config.json
No raw dataset or checkpoint is downloaded implicitly. ERA5 sources are read-only.
"""
from __future__ import annotations
from collections import deque
import argparse
import glob
import hashlib
import json
from pathlib import Path
import numpy as np

from .data import validate_timestamps, write_residual_archive, ResidualArchive
from .resources import check_process_memory
from .rollout import ClosureController, pack_sigma, unpack_sigma
from .training import load_checkpoint

VARIABLES = ('temperature','u','v','specific_humidity')
UNITS = {'temperature':'K','u':'m/s','v':'m/s','specific_humidity':'kg/kg'}
ALIASES = {'t':'temperature','q':'specific_humidity','z':'geopotential',
           'u':'u_component_of_wind','v':'v_component_of_wind',
           'sst':'sea_surface_temperature','siconc':'sea_ice_cover',
           'ci':'sea_ice_cover','clwc':'specific_cloud_liquid_water_content',
           'ciwc':'specific_cloud_ice_water_content'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def checkpoint_sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


class ERA5Reader:
    """Index timestamps only; load/regrid one record at a time."""
    def __init__(self, sources, backend, *, regrid=False):
        import xarray as xr
        self.xr = xr
        sources = [sources] if isinstance(sources,str) else sources
        self.paths = sorted({p for pattern in sources for p in glob.glob(pattern)})
        if not self.paths:
            raise FileNotFoundError(f'No ERA5 inputs match {sources}')
        self.index = []
        self.backend = backend
        self.regrid = regrid
        self._regridder = None
        self._source_grid = None
        for p in self.paths:
            with self.open(p) as ds:
                name = 'time' if 'time' in ds.coords else 'valid_time'
                for i,t in enumerate(np.asarray(ds[name].values).reshape(-1)):
                    self.index.append((np.datetime64(t,'ns'),p,i,name))
        self.index.sort(key=lambda item:item[0])
        self.times,_ = validate_timestamps([item[0] for item in self.index])

    def open(self,path):
        return self.xr.open_zarr(path,chunks=None) if path.endswith('.zarr') else self.xr.open_dataset(path)

    def record(self,index):
        time,path,i,time_name = self.index[index]
        with self.open(path) as ds:
            if time_name in ds.dims:
                ds = ds.isel({time_name:i})
            renames = {**ALIASES,'valid_time':'time','pressure_level':'level','lat':'latitude','lon':'longitude'}
            ds = ds.rename({a:b for a,b in renames.items() if a in ds and b not in ds})
            needed = self.backend.model.input_variables+self.backend.model.forcing_variables
            missing = set(needed)-set(ds.data_vars)
            if missing:
                raise ValueError(f'ERA5 is missing NeuralGCM inputs/forcings: {sorted(missing)}')
            ds = ds[needed].sortby('latitude').sortby('longitude')
            levels = np.asarray(self.backend.model.data_coords.vertical.centers)
            try:
                ds = ds.sel(level=levels)
            except KeyError as exc:
                raise ValueError('ERA5 must include every checkpoint pressure level, in hPa; no vertical interpolation is implicit') from exc
            ds = ds.load()
        if self.regrid:
            from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
            signature = (tuple(ds.latitude.values),tuple(ds.longitude.values))
            if self._source_grid is not None and signature != self._source_grid:
                raise ValueError('ERA5 horizontal grids changed between records')
            if self._regridder is None:
                source = spherical_harmonic.Grid(
                    latitude_nodes=ds.sizes['latitude'],longitude_nodes=ds.sizes['longitude'],
                    latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
                    longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude))
                self._regridder = horizontal_interpolation.ConservativeRegridder(
                    source,self.backend.model.data_coords.horizontal,skipna=True)
                self._source_grid = signature
            ds = xarray_utils.regrid(ds,self._regridder)
            ds = xarray_utils.fill_nan_with_nearest(ds)
        # Official API checks all coordinates, dimensions and variable names.
        inputs=self.backend.model.inputs_from_xarray(ds)
        forcings=self.backend.model.forcings_from_xarray(ds)
        return ds,inputs,forcings


def _backend(config):
    from .neuralgcm_backend import NeuralGCMBackend
    return NeuralGCMBackend.from_checkpoint(config['neuralgcm_checkpoint'])


def _variables(config,backend):
    variables=tuple(config.get('variables',VARIABLES))
    if len(set(variables)) != len(variables) or any(v not in UNITS for v in variables):
        raise ValueError('Choose unique temperature/u/v/specific_humidity channels')
    levels=backend.sigma_shape[0]
    names=[f'{v}:sigma_{k}' for v in variables for k in range(levels)]
    units=[UNITS[v] for v in variables for _ in range(levels)]
    return variables,names,units


def prepare(config):
    import jax
    backend=_backend(config)
    reader=ERA5Reader(config['era5'],backend,regrid=config.get('regrid',False))
    _,dt=validate_timestamps(reader.times,config.get('dt_hours',6)*3600)
    variables,names,units=_variables(config,backend)
    output=Path(config['archive'])
    temporary=Path(str(output)+'.preparation')
    temporary.mkdir(parents=True,exist_ok=False)
    n=len(reader.times)
    sigma,lon,lat=backend.sigma_shape
    shape=(len(names),lat,lon)
    truth=np.lib.format.open_memmap(temporary/'truth.npy',mode='w+',dtype='float32',shape=(n,*shape))
    predictions=np.lib.format.open_memmap(temporary/'baseline.npy',mode='w+',dtype='float32',shape=(n-1,*shape))
    rng=jax.random.PRNGKey(config.get('encoder_seed',0))
    for i in range(n):
        check_process_memory(config.get('memory_limit_gb',96)*1e9/1024**3)
        _,inputs,forcing=reader.record(i)
        state=backend.encode(inputs,forcing,rng_key=rng)
        truth[i]=pack_sigma(backend.sigma_fields(state),variables)
        if i<n-1:
            next_state=backend.advance_interval(state,forcing,dt)
            predictions[i]=pack_sigma(backend.sigma_fields(next_state),variables)
        if i%20==0: print(f'encoded interval {i}/{n-1}',flush=True)
    truth.flush(); predictions.flush()
    write_residual_archive(truth,predictions,reader.times,output,channel_names=names,
        channel_units=units,latitudes=np.rad2deg(backend.model.model_coords.horizontal.latitudes),
        dt_seconds=dt,splits=config.get('splits'),budget_gib=config.get('memory_limit_gb',96)*1e9/1024**3,
        provenance={'coordinate_system':'NeuralGCM native sigma nodal',
                    'neuralgcm_checkpoint':str(config['neuralgcm_checkpoint']),
                    'neuralgcm_sha256':checkpoint_sha256(config['neuralgcm_checkpoint']),
                    'encoder_seed':config.get('encoder_seed',0),
                    'variables':list(variables),'target':'encoded observed next state minus frozen teacher-forced next state',
                    'forcing':'interval-start persistence','dt_seconds':dt})
    # Keep preparation arrays only on explicit request; they are reproducible scratch.
    if not config.get('keep_preparation',False):
        del truth,predictions
        (temporary/'truth.npy').unlink(); (temporary/'baseline.npy').unlink(); temporary.rmdir()
    return {'archive':str(output),'intervals':n-1,'dt_hours':dt/3600,'channels':len(names)}


def _pressure_fields(decoded,variables):
    names={'u':'u_component_of_wind','v':'v_component_of_wind'}
    return np.concatenate([np.asarray(decoded[names.get(v,v)]).transpose(0,2,1) for v in variables],axis=0).astype(np.float32)


def forecast(config):
    import jax
    backend=_backend(config)
    checkpoint=load_checkpoint(config['closure_checkpoint'])
    variables,names,_=_variables(config,backend)
    if names != checkpoint['channel_names']:
        raise ValueError('Checkpoint closure channels/sigma levels do not match NeuralGCM')
    trained_hash=checkpoint.get('provenance',{}).get('neuralgcm_sha256')
    if not trained_hash or trained_hash!=checkpoint_sha256(config['neuralgcm_checkpoint']):
        raise ValueError('Closure must be trained from this exact NeuralGCM checkpoint (SHA256)')
    dt=float(checkpoint['dt_seconds']); h=checkpoint['config']['history_steps']
    trained_seed=checkpoint.get('provenance',{}).get('encoder_seed',0)
    if config.get('encoder_seed',trained_seed)!=trained_seed:
        raise ValueError('History encoder seed must match residual training provenance')
    days=float(config.get('days',40)); members=int(config.get('members',4))
    if days<=0 or members<1 or not np.isclose(days*86400/dt,round(days*86400/dt)):
        raise ValueError('Positive forecast days and members; duration must be an integer number of archive steps')
    daily=86400/dt
    if not np.isclose(daily,round(daily)) or not np.isclose(days,round(days)):
        raise ValueError('Daily output requires dt to divide one day and whole forecast days')
    reader=ERA5Reader(config['era5'],backend,regrid=config.get('regrid',False))
    issue=np.datetime64(config['issue_time'],'ns')
    split=config.get('evaluation_split')
    if split is not None:
        bounds=checkpoint.get('provenance',{}).get('split_time_bounds',{})
        if split not in {'validation','test'} or split not in bounds:
            raise ValueError('Declared evaluation requires recorded validation/test time bounds')
        first,last=map(lambda value:np.datetime64(value,'ns'),bounds[split])
        history_start=issue-np.timedelta64(round(h*dt*1e9),'ns')
        forecast_end=issue+np.timedelta64(round(days*86400*1e9),'ns')
        if history_start<first or forecast_end>last:
            raise ValueError('Forecast history and full horizon must stay inside the declared evaluation split')
    indices=np.flatnonzero(reader.times==issue)
    if len(indices)!=1 or indices[0]<h:
        raise ValueError('Need a unique issue time and H preceding observed timestamps')
    origin=int(indices[0])
    validate_timestamps(reader.times[origin-h:origin+1],dt)
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=False)
    sample_ds,_,_=reader.record(origin)
    plevels=np.asarray(backend.model.data_coords.vertical.centers)
    out_names=[f'{v}:{level:g}hPa' for v in variables for level in plevels]
    out_shape=(1,members,round(days),len(out_names),sample_ds.sizes['latitude'],sample_ds.sizes['longitude'])
    base_file=np.lib.format.open_memmap(output/'baseline.npy',mode='w+',dtype='float32',shape=out_shape)
    ens_file=np.lib.format.open_memmap(output/'ensemble.npy',mode='w+',dtype='float32',shape=out_shape)
    for member in range(members):
        # Historical residual proxy uses fixed encoder noise, matching preparation.
        states=[]; residuals=[]; previous_prediction=None
        encoder_key=jax.random.PRNGKey(trained_seed)
        for i in range(origin-h,origin+1):
            _,inputs,forcing=reader.record(i)
            current=backend.encode(inputs,forcing,rng_key=encoder_key)
            field=pack_sigma(backend.sigma_fields(current),variables)
            if previous_prediction is not None:
                states.append(field); residuals.append((field-previous_prediction)/dt)
            if i<origin:
                pred=backend.advance_interval(current,forcing,dt)
                previous_prediction=pack_sigma(backend.sigma_fields(pred),variables)
        # Matched base and corrected initial stochastic state; independent per member.
        initial=backend.encode(inputs,forcing,rng_key=jax.random.PRNGKey(config.get('seed',42)+member))
        states[-1]=pack_sigma(backend.sigma_fields(initial),variables)
        controller=ClosureController(checkpoint,np.stack(states),np.stack(residuals),
            seed=config.get('seed',42)+member,correction_scale=config.get('correction_scale',0.1),
            tendency_clip_std=config.get('tendency_clip_std',5),tile_columns=config.get('tile_columns',256),
            mode=config.get('mode','full'),memory_limit_gb=config.get('memory_limit_gb',96))
        base=initial; corrected=initial
        for step in range(round(days*86400/dt)):
            check_process_memory(config.get('memory_limit_gb',96)*1e9/1024**3)
            tendency=controller.tendency()
            correction=None if not np.any(tendency) else unpack_sigma(tendency,variables,backend.sigma_shape[0])
            # The same issue-time SST/sea-ice is persisted for both trajectories.
            base=backend.advance_interval(base,forcing,dt)
            corrected=backend.advance_interval(corrected,forcing,dt,correction)
            controller.observe_generated(pack_sigma(backend.sigma_fields(corrected),variables),tendency)
            if (step+1)%round(daily)==0:
                day=(step+1)//round(daily)
                b=_pressure_fields(backend.decode(base,backend.persisted_forcings(base,forcing)),variables)
                e=_pressure_fields(backend.decode(corrected,backend.persisted_forcings(corrected,forcing)),variables)
                if not np.isfinite(b).all() or not np.isfinite(e).all():
                    raise FloatingPointError(f'Nonfinite decoded forecast at day {day}')
                base_file[0,member,day-1]=b; ens_file[0,member,day-1]=e
                base_file.flush(); ens_file.flush()
                print(f'member {member+1}/{members}, day {day}/{round(days)}',flush=True)
    latitudes=np.asarray(sample_ds.latitude.values)
    np.save(output/'latitudes.npy',latitudes)
    # The checkpoint data grid uses Gaussian nodes; preserve exact area quadrature.
    grid=backend.model.data_coords.horizontal
    weights=np.asarray(grid.quadrature_weights) if hasattr(grid,'quadrature_weights') else None
    if weights is not None and weights.ndim==2 and weights.shape[-1]==len(latitudes):
        if not np.allclose(weights,weights[0:1,:]):
            raise ValueError('Expected longitude-uniform quadrature weights')
        weights=weights[0,:]
    if weights is not None and weights.shape==(len(latitudes),):
        np.save(output/'latitude_weights.npy',weights)
    elif getattr(grid,'latitude_spacing',None)=='gauss':
        np.save(output/'latitude_weights.npy',np.polynomial.legendre.leggauss(len(latitudes))[1])
    metadata={'issue_time':str(issue),'lead_days':list(range(1,round(days)+1)),
        'channel_names':out_names,'variables':list(variables),'pressure_levels_hpa':plevels.tolist(),
        'members':members,'forcing':'issue-time persistence; no future ERA5 conditioning',
        'first_interval':'pure NeuralGCM','closure_mode':config.get('mode','full'),
        'correction_scale':config.get('correction_scale',0.1),'seed':config.get('seed',42),
        'neuralgcm_checkpoint':str(config['neuralgcm_checkpoint']),
        'closure_checkpoint':str(config['closure_checkpoint']),'status':'forecast only; skill unverified'}
    metadata['evaluation_split']=split
    metadata['split_bounds_checked']=split is not None
    write_json(output/'forecast.json',metadata)
    return metadata


def verify(config):
    """Read future truth ONLY here, after forecasting; climatology supplied separately."""
    from .metrics import evaluate_forecast_files
    output=Path(config['forecast_output'])
    metadata=json.loads((output/'forecast.json').read_text())
    if not metadata.get('split_bounds_checked',False):
        raise ValueError('Skill verification requires a forecast with a checked validation/test evaluation_split')
    if metadata.get('evaluation_split') is not None and config.get('evaluation_split','test')!=metadata['evaluation_split']:
        raise ValueError('Verification split disagrees with the checked forecast split')
    backend=_backend(config)
    reader=ERA5Reader(config['era5'],backend,regrid=config.get('regrid',False))
    issue=np.datetime64(metadata['issue_time'],'ns')
    ensemble=np.load(output/'ensemble.npy',mmap_mode='r')
    truth=np.lib.format.open_memmap(output/'truth.npy',mode='w+',dtype='float32',shape=(1,*ensemble.shape[2:]))
    for lead,day in enumerate(metadata['lead_days']):
        target=issue+np.timedelta64(day,'D')
        indices=np.flatnonzero(reader.times==target)
        if len(indices)!=1: raise ValueError(f'Missing verification timestamp {target}')
        _,inputs,_=reader.record(int(indices[0]))
        truth[0,lead]=_pressure_fields(inputs,metadata['variables'])
    truth.flush()
    weight_path=output/'latitude_weights.npy'
    result=evaluate_forecast_files(output/'baseline.npy',output/'ensemble.npy',output/'truth.npy',
        config['climatology'],lead_days=metadata['lead_days'],channel_names=metadata['channel_names'],
        latitudes=np.load(output/'latitudes.npy'),latitude_weights=np.load(weight_path) if weight_path.exists() else None,
        short_tolerance=config.get('short_tolerance',0.02),
        evaluation_split=config.get('evaluation_split','test'),
        climatology_provenance=config['climatology_provenance'],
        budget_gib=config.get('memory_limit_gb',96)*1e9/1024**3)
    write_json(output/'evaluation.json',result)
    return result


def climatology(config):
    """Annual mean pressure climatology using exactly the archive train period."""
    backend=_backend(config)
    reader=ERA5Reader(config['era5'],backend,regrid=config.get('regrid',False))
    archive=ResidualArchive(config['archive'])
    start,stop=archive.splits['train']
    train_times=archive.timestamps[start:stop]
    indices=np.searchsorted(reader.times,train_times)
    if np.any(indices>=len(reader.times)) or not np.array_equal(reader.times[indices],train_times):
        raise ValueError('ERA5 does not cover the exact archive training timestamps')
    variables,names,_=_variables(config,backend)
    provenance=archive.manifest.get('provenance',{})
    expected=checkpoint_sha256(config['neuralgcm_checkpoint'])
    if provenance.get('neuralgcm_sha256')!=expected:
        raise ValueError('Climatology and archive must use the same NeuralGCM grid/checkpoint')
    mean=None
    for count,index in enumerate(indices,1):
        check_process_memory(config.get('memory_limit_gb',96)*1e9/1024**3)
        _,inputs,_=reader.record(int(index))
        values=_pressure_fields(inputs,variables).astype(np.float64)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite climatology input')
        mean=values if mean is None else mean+(values-mean)/count
    output=Path(config['climatology']); output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    np.save(output,mean.astype(np.float32))
    levels=np.asarray(backend.model.data_coords.vertical.centers)
    metadata={'fit_split':'train','reference':'annual mean (not day-of-year climatology)',
              'train_start':str(train_times[0]),'train_end':str(train_times[-1]),
              'samples':len(indices),'neuralgcm_sha256':expected,
              'channel_names':[f'{v}:{level:g}hPa' for v in variables for level in levels]}
    write_json(output.with_suffix('.json'),metadata)
    return metadata


def run(config):
    """One command for a complete, explicitly configured research experiment."""
    from .training import TrainingConfig, train
    shared={k:config[k] for k in ('neuralgcm_checkpoint','era5','regrid','memory_limit_gb') if k in config}
    root=Path(config['run_directory'])
    root.mkdir(parents=True,exist_ok=False)
    prep={**shared,'archive':str(root/'archive'),**config.get('prepare',{})}
    prepare(prep)
    model_path=root/'closure.npz'
    train(prep['archive'],model_path,TrainingConfig(**config.get('train',{})))
    climate_path=root/'climatology.npy'
    variable_config={'variables':prep.get('variables',list(VARIABLES))}
    climatology({**shared,**variable_config,'archive':prep['archive'],'climatology':str(climate_path)})
    results=[]
    for i,case in enumerate(config['cases']):
        prediction={**shared,**variable_config,**case,'evaluation_split':case.get('evaluation_split','validation'),
                    'closure_checkpoint':str(model_path),'output':str(root/f'case_{i:03d}')}
        forecast(prediction)
        verification={**shared,'forecast_output':prediction['output'],'climatology':str(climate_path),
            'evaluation_split':case.get('evaluation_split','validation'),
            'climatology_provenance':str(climate_path.with_suffix('.json')),
            'short_tolerance':config.get('short_tolerance',0.)}
        report=verify(verification)
        results.append({'forecast':prediction['output'],'evaluation_split':verification['evaluation_split'],
                        'short_range_gate':report['short_range_gate']})
    return {'run_directory':str(root),'cases':results,'skill_claim':'See actual evaluation reports; no automatic improvement claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','forecast','verify','climatology','run'])
    parser.add_argument('config')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    print(json.dumps(globals()[args.command](config),indent=2,allow_nan=False))


if __name__=='__main__':
    main()
