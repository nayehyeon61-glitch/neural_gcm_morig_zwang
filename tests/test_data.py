import json

import numpy as np
import pytest

from neuralgcm_mz.data import (ChannelNormalizer, ResidualArchive,
                              fit_channel_normalizer, iter_era5_records,
                              iter_windows, validate_timestamps,
                              write_residual_archive)
from neuralgcm_mz.resources import memory_preflight


def make_archive(tmp_path):
    n = 30
    times = np.datetime64("2000-01-01") + np.arange(n+1)*np.timedelta64(6, "h")
    truth = np.arange(n+1, dtype=np.float32)[:, None, None, None] * np.ones((1, 2, 3, 4), dtype=np.float32)
    baseline = truth[1:] - 2
    # Strong distribution shift outside training must not contaminate statistics.
    baseline[18:] -= 100
    path = tmp_path / "archive"
    write_residual_archive(truth, baseline, times, path, channel_names=["T", "q"],
                           latitudes=[-60, 0, 60], channel_units=["K", "kg kg^-1"],
                           splits={"train": [0, 18], "validation": [18, 24], "test": [24, 30]},
                           chunk_size=3)
    return ResidualArchive(path), truth


def test_residual_sign_units_and_causal_windows(tmp_path):
    archive, truth = make_archive(tmp_path)
    assert isinstance(archive.residual, np.memmap)
    np.testing.assert_allclose(archive.residual[0], 2/(6*3600))
    window = next(iter_windows(archive, "validation", history_steps=2, future_steps=3))
    assert window["origin_index"] == 20
    np.testing.assert_array_equal(window["state_history"], truth[19:21])
    np.testing.assert_array_equal(window["residual_history"], archive.residual[18:20])
    np.testing.assert_array_equal(window["residual_future"], archive.residual[20:23])
    assert list(archive.window_origins("validation", 2, 3)) == [20, 21]
    assert not window["residual_future"].flags.writeable


def test_normalizer_training_only_and_round_trip(tmp_path):
    archive, _ = make_archive(tmp_path)
    norm = fit_channel_normalizer(archive, source="states", chunk_size=4)
    expected = archive.states[:18].mean(axis=(0, 2, 3))
    np.testing.assert_allclose(norm.mean, expected)
    residual_norm = fit_channel_normalizer(archive, chunk_size=5)
    np.testing.assert_allclose(residual_norm.mean, 2/21600)
    assert np.all(residual_norm.std > 0)
    path = tmp_path / "normalizer.json"
    norm.save(path)
    loaded = ChannelNormalizer.load(path)
    np.testing.assert_allclose(loaded.inverse(loaded.transform(archive.states[:2])), archive.states[:2], atol=1e-6)
    payload = json.loads(path.read_text())
    payload["fit_split"] = "test"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training"):
        ChannelNormalizer.load(path)


def test_timestamp_gaps_and_mismatched_cadence_rejected():
    with pytest.raises(ValueError, match="regular"):
        validate_timestamps(["2000-01-01", "2000-01-02", "2000-01-04"])
    with pytest.raises(ValueError, match="cadence"):
        validate_timestamps(["2000-01-01", "2000-01-02"], 6*3600)


def test_splits_and_memory_preflight(tmp_path):
    archive, _ = make_archive(tmp_path)
    assert len(list(archive.window_origins("test", 4, 4))) == 0
    plan = memory_preflight(channels=8, latitude=32, longitude=64,
                            history_steps=8, future_steps=8, budget_gib=1)
    assert plan.estimated_bytes < plan.usable_bytes < plan.budget_bytes
    with pytest.raises(MemoryError):
        memory_preflight(channels=100, latitude=721, longitude=1440,
                         history_steps=100, future_steps=100)
    with pytest.raises(ValueError, match="96"):
        memory_preflight(channels=1, latitude=1, longitude=1,
                         history_steps=1, future_steps=1, budget_gib=100)


def test_era5_records_stream_real_timestamps(tmp_path):
    xr = pytest.importorskip("xarray")
    time = np.datetime64("2001-01-01") + np.arange(4)*np.timedelta64(6, "h")
    values = np.arange(4*2*3*4).reshape(4, 2, 3, 4)
    ds = xr.Dataset({"t": (("time", "level", "latitude", "longitude"), values)},
                    coords={"time": time, "level": [500, 850],
                            "latitude": [-60, 0, 60], "longitude": [0, 90, 180, 270]})
    path = tmp_path / "era5.nc"
    ds.to_netcdf(path)
    records = list(iter_era5_records([path], [{"name": "t850", "variable": "t", "level": 850}]))
    assert len(records) == 4
    np.testing.assert_array_equal(records[2]["fields"][0], values[2, 1])
    assert records[2]["time"] == time[2]
