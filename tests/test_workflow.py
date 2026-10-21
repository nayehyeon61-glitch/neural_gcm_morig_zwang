"""Small contract tests for I/O boundaries, independent of a real checkpoint."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from neuralgcm_mz import workflow
from neuralgcm_mz.data import ResidualArchive


def test_reader_maps_era5_wind_names_and_preserves_time(tmp_path):
    xr = pytest.importorskip("xarray")
    calls = []
    required = ["temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity", "geopotential"]
    def inputs(ds):
        calls.append(ds)
        assert all(name in ds for name in required)
        return {name: ds[name].transpose("level", "longitude", "latitude").values for name in required}
    model = SimpleNamespace(input_variables=required,
        forcing_variables=["sea_surface_temperature", "sea_ice_cover"],
        data_coords=SimpleNamespace(vertical=SimpleNamespace(centers=np.array([500., 850.]))),
        inputs_from_xarray=inputs,
        forcings_from_xarray=lambda ds: {"time": ds.time.values})
    time = np.datetime64("2001-01-01") + np.arange(3)*np.timedelta64(6, "h")
    shape = (3, 2, 2, 2)
    ds = xr.Dataset({name: (("valid_time", "pressure_level", "lat", "lon"), np.ones(shape)*i)
                     for i, name in enumerate(["t", "u", "v", "q", "z"])},
                     coords={"valid_time": time, "pressure_level": [850., 500.],
                             "lat": [30., -30.], "lon": [180., 0.]})
    ds["sst"] = (("valid_time", "lat", "lon"), np.ones((3, 2, 2))*280)
    ds["ci"] = (("valid_time", "lat", "lon"), np.zeros((3, 2, 2)))
    path = tmp_path/"era5.nc"
    ds.to_netcdf(path)
    reader = workflow.ERA5Reader(str(path), SimpleNamespace(model=model))
    record, _, forcings = reader.record(1)
    np.testing.assert_array_equal(record.latitude, [-30, 30])
    np.testing.assert_array_equal(record.level, [500, 850])
    assert np.datetime64(forcings["time"]) == time[1]


def test_pressure_output_maps_official_wind_keys():
    fields = {"temperature": np.ones((2, 3, 4)),
              "u_component_of_wind": np.full((2, 3, 4), 2),
              "v_component_of_wind": np.full((2, 3, 4), 3),
              "specific_humidity": np.full((2, 3, 4), 4)}
    result = workflow._pressure_fields(fields, ("temperature", "u", "v", "specific_humidity"))
    assert result.shape == (8, 4, 3)
    np.testing.assert_array_equal(result[:, 0, 0], [1, 1, 2, 2, 3, 3, 4, 4])


def test_reader_accepts_scalar_time_coordinates_across_files(tmp_path):
    xr = pytest.importorskip("xarray")
    times = np.datetime64("2001-01-01") + np.arange(2)*np.timedelta64(1, "D")
    for i, time in enumerate(times):
        ds = xr.Dataset({"t": (("pressure_level", "lat", "lon"), np.full((1, 2, 2), i))},
                         coords={"valid_time": time, "pressure_level": [500.],
                                 "lat": [-30., 30.], "lon": [0., 180.]})
        ds.to_netcdf(tmp_path/f"day_{i}.nc")
    model = SimpleNamespace(input_variables=["temperature"], forcing_variables=[],
        data_coords=SimpleNamespace(vertical=SimpleNamespace(centers=np.array([500.]))),
        inputs_from_xarray=lambda ds: {"temperature": ds.temperature.values},
        forcings_from_xarray=lambda ds: {"time": ds.time.values})
    reader = workflow.ERA5Reader(str(tmp_path/"day_*.nc"), SimpleNamespace(model=model))
    assert len(reader.times) == 2
    record, fields, forcing = reader.record(1)
    assert "time" not in record.dims
    assert np.datetime64(forcing["time"]) == times[1]
    np.testing.assert_array_equal(fields["temperature"], 1)


class FakeBackend:
    sigma_shape = (1, 2, 2)
    def __init__(self):
        grid = SimpleNamespace(latitudes=np.deg2rad([-30, 30]), latitude_spacing="gauss",
                               quadrature_weights=np.array([[1., 1.], [1., 1.]]))
        self.model = SimpleNamespace(model_coords=SimpleNamespace(horizontal=grid),
                                    data_coords=SimpleNamespace(horizontal=grid,
                                        vertical=SimpleNamespace(centers=np.array([500.]))))
        self.forcing_times = []
    def encode(self, inputs, forcings, rng_key=None):
        return inputs["temperature"].copy()
    def sigma_fields(self, state):
        return {"temperature": state}
    def advance_interval(self, state, forcing, dt, correction=None):
        self.forcing_times.append(forcing["time_index"])
        return state+dt/86400+(0 if correction is None else correction["temperature"])
    def decode(self, state, forcing):
        return {"temperature": state}
    def persisted_forcings(self, state, forcing):
        return forcing
    def spherical_noise(self, length_km):
        from neuralgcm_mz.spatial import independent_noise
        return independent_noise(2, 2)
    timestep_seconds = 3600.
    def with_memory(self, state, previous):
        return state + 0.5*(previous-state)  # visible memory effect for the test


class FakeReader:
    visited = []
    max_allowed = None
    def __init__(self, sources, backend, regrid=False):
        self.times = np.datetime64("2000-01-01") + np.arange(12)*np.timedelta64(1, "D")
    def record(self, index):
        import xarray as xr
        if self.max_allowed is not None and index > self.max_allowed:
            raise AssertionError("Forecast attempted to read future observations")
        self.visited.append(index)
        ds = xr.Dataset(coords={"latitude": [-30., 30.], "longitude": [0., 180.]})
        return ds, {"temperature": np.full((1, 2, 2), index, dtype=np.float32)}, {"time_index": index}


class FakeController:
    def __init__(self, checkpoint, states, past, **kwargs):
        assert states.shape == past.shape == (2, 1, 2, 2)
        self.steps = 0
    def tendency(self):
        return np.zeros((1, 2, 2), dtype=np.float32) if not self.steps else np.full((1, 2, 2), 0.1, dtype=np.float32)
    def observe_generated(self, state, tendency):
        self.steps += 1


def test_forecast_reads_history_only_and_first_interval_is_base(tmp_path, monkeypatch):
    backend = FakeBackend()
    FakeReader.visited = []
    FakeReader.max_allowed = 2
    monkeypatch.setattr(workflow, "_backend", lambda config: backend)
    monkeypatch.setattr(workflow, "ERA5Reader", FakeReader)
    monkeypatch.setattr(workflow, "ClosureController", FakeController)
    base_checkpoint = tmp_path/"trusted.pkl"
    base_checkpoint.write_bytes(b"fake-backend-checkpoint-identity")
    monkeypatch.setattr(workflow, "load_checkpoint", lambda path: {
        "channel_names": ["temperature:sigma_0"], "dt_seconds": 86400,
        "config": {"history_steps": 2},
        "provenance": {"neuralgcm_sha256": workflow.checkpoint_sha256(base_checkpoint)}})
    output = tmp_path/"forecast"
    workflow.forecast({"neuralgcm_checkpoint": str(base_checkpoint), "closure_checkpoint": "model.npz",
                       "variables": ["temperature"], "era5": "dummy", "issue_time": "2000-01-03",
                       "output": str(output), "days": 5, "members": 2})
    assert max(FakeReader.visited) == 2
    baseline = np.load(output/"baseline.npy")
    ensemble = np.load(output/"ensemble.npy")
    assert baseline.shape == ensemble.shape == (1, 2, 5, 1, 2, 2)
    np.testing.assert_array_equal(baseline[:, :, 0], ensemble[:, :, 0])
    assert np.all(ensemble[:, :, -1] > baseline[:, :, -1])
    assert (output/"latitude_weights.npy").exists()


def test_prepare_produces_sigma_residuals_with_correct_units(tmp_path, monkeypatch):
    backend = FakeBackend()
    FakeReader.visited = []
    FakeReader.max_allowed = None
    monkeypatch.setattr(workflow, "_backend", lambda config: backend)
    monkeypatch.setattr(workflow, "ERA5Reader", FakeReader)
    base_checkpoint = tmp_path/"trusted.pkl"
    base_checkpoint.write_bytes(b"fake-backend-checkpoint-identity")
    path = tmp_path/"archive"
    workflow.prepare({"neuralgcm_checkpoint": str(base_checkpoint), "variables": ["temperature"],
                      "era5": "dummy", "archive": str(path), "dt_hours": 24})
    archive = ResidualArchive(path)
    assert archive.states.shape == (11, 1, 2, 2)
    np.testing.assert_array_equal(archive.residual, 0)
    assert archive.manifest["residual_units"] == ["(K) s^-1"]
    assert archive.manifest["provenance"]["coordinate_system"] == "NeuralGCM native sigma nodal"
    assert archive.manifest["provenance"]["neuralgcm_sha256"] == workflow.checkpoint_sha256(base_checkpoint)


def test_forecast_history_follows_archive_memory_mode(tmp_path, monkeypatch):
    backend = FakeBackend()
    calls = []
    backend.with_memory = lambda state, previous: calls.append(1) or state
    FakeReader.max_allowed = 2
    monkeypatch.setattr(workflow, "_backend", lambda config: backend)
    monkeypatch.setattr(workflow, "ERA5Reader", FakeReader)
    monkeypatch.setattr(workflow, "ClosureController", FakeController)
    base_checkpoint = tmp_path/"trusted.pkl"
    base_checkpoint.write_bytes(b"fake-backend-checkpoint-identity")
    monkeypatch.setattr(workflow, "load_checkpoint", lambda path: {
        "channel_names": ["temperature:sigma_0"], "dt_seconds": 86400, "config": {"history_steps": 2},
        "provenance": {"neuralgcm_sha256": workflow.checkpoint_sha256(base_checkpoint), "memory": "warm (x)"}})
    workflow.forecast({"neuralgcm_checkpoint": str(base_checkpoint), "closure_checkpoint": "model.npz",
                       "variables": ["temperature"], "era5": "dummy", "issue_time": "2000-01-03",
                       "output": str(tmp_path/"forecast"), "days": 2, "members": 1})
    assert len(calls) == 1  # history intervals 0 (cold, no past) and 1 (warm)
    meta = json.loads((tmp_path/"forecast"/"forecast.json").read_text())
    assert meta["history_memory"] == "warm"


def test_prepare_warm_memory_uses_previous_trajectory(tmp_path, monkeypatch):
    backend = FakeBackend()
    FakeReader.max_allowed = None
    monkeypatch.setattr(workflow, "_backend", lambda config: backend)
    monkeypatch.setattr(workflow, "ERA5Reader", FakeReader)
    base_checkpoint = tmp_path/"trusted.pkl"
    base_checkpoint.write_bytes(b"fake-backend-checkpoint-identity")
    path = tmp_path/"archive"
    workflow.prepare({"neuralgcm_checkpoint": str(base_checkpoint), "variables": ["temperature"],
                      "era5": "dummy", "archive": str(path), "dt_hours": 24, "warm_memory": True})
    archive = ResidualArchive(path)
    np.testing.assert_array_equal(archive.states[:, 0, 0, 0], np.arange(11))  # truth is the encoding
    # Interval 0 has no past and stays cold. Interval 1 starts from the
    # encoding 1 blended with the previous trajectory one native step before
    # its end (23/24), i.e. 1 - 1/48, so its 1-day residual is 1/48 per day.
    assert archive.residual[0, 0, 0, 0] == 0
    np.testing.assert_allclose(archive.residual[1, 0, 0, 0], 1/48/86400, rtol=1e-5)
    assert archive.manifest["provenance"]["memory"].startswith("warm")
    assert len(backend.forcing_times) == 2 * 11  # two legs per interval


def test_forecast_rejects_wrong_backbone_checkpoint_before_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_backend", lambda config: FakeBackend())
    monkeypatch.setattr(workflow, "load_checkpoint", lambda path: {
        "channel_names": ["temperature:sigma_0"], "provenance": {"neuralgcm_sha256": "wrong-hash"}})
    checkpoint = tmp_path/"base.pkl"
    checkpoint.write_bytes(b"identity")
    with pytest.raises(ValueError, match="SHA256"):
        workflow.forecast({"neuralgcm_checkpoint": str(checkpoint), "closure_checkpoint": "closure.npz",
                           "variables": ["temperature"]})


@pytest.mark.parametrize("bounds", [("2000-01-02", "2000-01-12"),
                                    ("2000-01-01", "2000-01-07")])
def test_declared_split_rejects_history_or_horizon_outside_bounds(tmp_path, monkeypatch, bounds):
    FakeReader.visited = []
    monkeypatch.setattr(workflow, "_backend", lambda config: FakeBackend())
    monkeypatch.setattr(workflow, "ERA5Reader", FakeReader)
    checkpoint = tmp_path/"base.pkl"
    checkpoint.write_bytes(b"identity")
    monkeypatch.setattr(workflow, "load_checkpoint", lambda path: {
        "channel_names": ["temperature:sigma_0"], "dt_seconds": 86400,
        "config": {"history_steps": 2},
        "provenance": {"neuralgcm_sha256": workflow.checkpoint_sha256(checkpoint),
                       "split_time_bounds": {"test": bounds}}})
    output = tmp_path/"forecast"
    with pytest.raises(ValueError, match="declared evaluation split"):
        workflow.forecast({"neuralgcm_checkpoint": str(checkpoint), "closure_checkpoint": "closure.npz",
                           "variables": ["temperature"], "era5": "dummy", "issue_time": "2000-01-03",
                           "evaluation_split": "test", "days": 5, "output": str(output)})
    assert FakeReader.visited == []
    assert not output.exists()


def test_climatology_reads_training_period_only(tmp_path, monkeypatch):
    backend = FakeBackend()
    FakeReader.visited = []
    FakeReader.max_allowed = None
    monkeypatch.setattr(workflow, "_backend", lambda config: backend)
    monkeypatch.setattr(workflow, "ERA5Reader", FakeReader)
    checkpoint = tmp_path/"base.pkl"
    checkpoint.write_bytes(b"fake-backend-checkpoint-identity")
    archive_path = tmp_path/"archive"
    config = {"neuralgcm_checkpoint": str(checkpoint), "variables": ["temperature"],
              "era5": "dummy", "archive": str(archive_path), "dt_hours": 24}
    workflow.prepare(config)
    archive = ResidualArchive(archive_path)
    start, stop = archive.splits["train"]
    FakeReader.visited = []
    FakeReader.max_allowed = stop-1
    output = tmp_path/"climatology.npy"
    metadata = workflow.climatology({**config, "climatology": str(output)})
    assert metadata["fit_split"] == "train"
    assert metadata["samples"] == stop-start
    assert FakeReader.visited == list(range(start, stop))
    np.testing.assert_array_equal(np.load(output), np.mean(np.arange(start, stop)))
    FakeReader.max_allowed = None


def test_forecast_rejects_negative_noise_correlation(tmp_path, monkeypatch):
    backend = FakeBackend()
    FakeReader.max_allowed = None
    monkeypatch.setattr(workflow, "_backend", lambda config: backend)
    monkeypatch.setattr(workflow, "ERA5Reader", FakeReader)
    monkeypatch.setattr(workflow, "ClosureController", FakeController)
    base_checkpoint = tmp_path/"trusted.pkl"
    base_checkpoint.write_bytes(b"fake-backend-checkpoint-identity")
    monkeypatch.setattr(workflow, "load_checkpoint", lambda path: {
        "channel_names": ["temperature:sigma_0"], "dt_seconds": 86400,
        "config": {"history_steps": 2},
        "provenance": {"neuralgcm_sha256": workflow.checkpoint_sha256(base_checkpoint)}})
    with pytest.raises(ValueError, match="noise_correlation_km"):
        workflow.forecast({"neuralgcm_checkpoint": str(base_checkpoint), "closure_checkpoint": "model.npz",
                           "variables": ["temperature"], "era5": "dummy", "issue_time": "2000-01-03",
                           "output": str(tmp_path/"forecast"), "days": 5, "members": 1,
                           "noise_correlation_km": -500})
