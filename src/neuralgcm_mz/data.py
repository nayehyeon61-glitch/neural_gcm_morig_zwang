"""Streaming residual archives with explicit causal and chronological alignment.

Archive sample ``i`` stores the resolved state at t_i and the correction
``(truth(t_i+dt) - NeuralGCM(truth(t_i), dt))/dt``. Training may use future
observations as targets; an autonomous rollout must never read them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from .resources import MAX_HOST_GIB, check_process_memory


def validate_timestamps(timestamps, dt_seconds: float | None = None) -> tuple[np.ndarray, float]:
    times = np.asarray(timestamps, dtype="datetime64[ns]")
    if times.ndim != 1 or len(times) < 2 or np.isnat(times).any():
        raise ValueError("At least two valid one-dimensional timestamps required")
    delta_ns = np.diff(times).astype("timedelta64[ns]").astype(np.int64)
    if np.any(delta_ns <= 0) or np.any(delta_ns != delta_ns[0]):
        raise ValueError("Timestamps must be strictly increasing and regularly sampled; no interpolation")
    actual = float(delta_ns[0]) / 1e9
    if dt_seconds is not None and not np.isclose(actual, dt_seconds, rtol=0, atol=1e-9):
        raise ValueError(f"Timestamp cadence {actual}s differs from dt_seconds={dt_seconds}")
    return times, actual


def chronological_splits(n: int, train_fraction: float = 0.7,
                         validation_fraction: float = 0.15) -> dict[str, list[int]]:
    if n < 3 or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Three nonempty chronological splits are required")
    a = int(n * train_fraction)
    b = int(n * (train_fraction + validation_fraction))
    if not 0 < a < b < n:
        raise ValueError("Too few samples for requested split fractions")
    return {"train": [0, a], "validation": [a, b], "test": [b, n]}


def _validate_splits(splits: Mapping[str, Sequence[int]], n: int) -> dict[str, list[int]]:
    normalized = {}
    previous_end = 0
    for name in ("train", "validation", "test"):
        if name not in splits or len(splits[name]) != 2:
            raise ValueError("Provide train, validation and test [start,stop) indices")
        start, stop = map(int, splits[name])
        if not 0 <= start < stop <= n or start < previous_end:
            raise ValueError("Splits must be nonempty, disjoint and chronological")
        normalized[name] = [start, stop]
        previous_end = stop
    return normalized


def write_residual_archive(truth, baseline_next, timestamps, path, *,
                           channel_names: Sequence[str], latitudes,
                           channel_units: Sequence[str] | None = None,
                           splits: Mapping[str, Sequence[int]] | None = None,
                           dt_seconds: float | None = None, chunk_size: int = 1,
                           budget_gib: float = MAX_HOST_GIB,
                           provenance: Mapping | None = None) -> Path:
    """Stream arrays into a new archive without loading a multi-year dataset.

    ``truth`` has shape [N+1,C,Y,X], ``baseline_next`` [N,C,Y,X], and
    ``timestamps`` [N+1]. Baselines must each start from *that interval's*
    true initial state (teacher forcing), using the frozen NeuralGCM model.
    Resolved model/sigma state channels, their units and grid must match.
    Existing archives are refused to avoid accidental replacement.
    """
    times, cadence = validate_timestamps(timestamps, dt_seconds)
    if len(truth.shape) != 4 or tuple(baseline_next.shape) != (truth.shape[0] - 1, *truth.shape[1:]):
        raise ValueError("Expected truth [N+1,C,Y,X] and baseline_next [N,C,Y,X]")
    n, c, ny, nx = baseline_next.shape
    latitudes = np.asarray(latitudes, dtype=np.float64)
    if len(times) != n + 1 or len(channel_names) != c or latitudes.shape != (ny,):
        raise ValueError("Timestamp, channel or latitude dimensions do not match")
    if not np.isfinite(latitudes).all() or np.any(np.abs(latitudes) > 90):
        raise ValueError("Latitudes must be finite degrees in [-90,90]")
    if len(set(channel_names)) != c or chunk_size < 1:
        raise ValueError("Unique channel names and a positive chunk size required")
    units = list(channel_units or ["model_state_unit"] * c)
    if len(units) != c:
        raise ValueError("Provide one state unit per channel")
    split_map = _validate_splits(splits or chronological_splits(n), n)
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Archive directory is not empty: {output}")
    shape = (n, c, ny, nx)
    states = np.lib.format.open_memmap(output / "states.npy", mode="w+", dtype="float32", shape=shape)
    residual = np.lib.format.open_memmap(output / "residual.npy", mode="w+", dtype="float32", shape=shape)
    for start in range(0, n, chunk_size):
        check_process_memory(budget_gib)
        stop = min(n, start + chunk_size)
        x = np.asarray(truth[start:stop], dtype=np.float32)
        y = np.asarray(truth[start + 1:stop + 1], dtype=np.float32)
        pred = np.asarray(baseline_next[start:stop], dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(pred).all():
            raise ValueError(f"Non-finite truth or baseline at samples {start}:{stop}")
        states[start:stop] = x
        residual[start:stop] = (y - pred) / cadence
    states.flush()
    residual.flush()
    np.save(output / "timestamps.npy", times[:-1])
    np.save(output / "latitudes.npy", latitudes)
    manifest = {"schema_version": 1, "shape": list(shape), "dt_seconds": cadence,
                "channel_names": list(channel_names), "state_units": units,
                "residual_units": [f"({u}) s^-1" for u in units],
                "splits": split_map, "alignment": "state_at_t; residual_for_t_to_t_plus_dt",
                "baseline_mode": "one_step_teacher_forced", "provenance": dict(provenance or {})}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    check_process_memory(budget_gib)
    return output


class ResidualArchive:
    """Read-only memmap-backed archive; callers can use arrays directly."""
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported residual archive schema")
        if self.manifest.get("alignment") != "state_at_t; residual_for_t_to_t_plus_dt":
            raise ValueError("Residual archive has an unknown time alignment")
        self.states = np.load(self.path / "states.npy", mmap_mode="r", allow_pickle=False)
        self.residual = np.load(self.path / "residual.npy", mmap_mode="r", allow_pickle=False)
        self.timestamps, self.dt_seconds = validate_timestamps(
            np.load(self.path / "timestamps.npy", allow_pickle=False), self.manifest["dt_seconds"])
        self.latitudes = np.load(self.path / "latitudes.npy", allow_pickle=False)
        self.channel_names = self.manifest["channel_names"]
        expected = tuple(self.manifest["shape"])
        if self.states.shape != expected or self.residual.shape != expected or len(expected) != 4:
            raise ValueError("Archive arrays do not match manifest shape")
        if len(self.timestamps) != expected[0] or len(self.channel_names) != expected[1] or self.latitudes.shape != (expected[2],):
            raise ValueError("Archive coordinates do not match arrays")
        self.splits = _validate_splits(self.manifest["splits"], expected[0])

    def window_origins(self, split: str, history_steps: int, future_steps: int,
                       stride: int = 1) -> range:
        """All history and targets stay inside the requested split."""
        if min(history_steps, future_steps, stride) < 1:
            raise ValueError("Window sizes and stride must be positive")
        start, stop = self.splits[split]
        # residual_history ends at origin-1; state_history includes origin.
        return range(start + history_steps, stop - future_steps + 1, stride)


def iter_windows(archive: ResidualArchive, split: str, history_steps: int,
                 future_steps: int, stride: int = 1) -> Iterator[dict]:
    """Yield views with no future data in conditioning.

    Dict: state_history [H,C,Y,X] at t_(j-H+1)..t_j;
    residual_history [H,C,Y,X] for intervals j-H..j-1;
    residual_future [F,C,Y,X] for intervals j..j+F-1;
    origin_index integer j and origin_time datetime64. Views are read-only.
    """
    for j in archive.window_origins(split, history_steps, future_steps, stride):
        yield {"state_history": archive.states[j-history_steps+1:j+1],
               "residual_history": archive.residual[j-history_steps:j],
               "residual_future": archive.residual[j:j+future_steps],
               "origin_index": j, "origin_time": archive.timestamps[j]}


@dataclass
class ChannelNormalizer:
    mean: np.ndarray
    std: np.ndarray
    count: int
    source: str
    train_start: int
    train_stop: int

    def transform(self, values):
        return (values - self.mean[:, None, None]) / self.std[:, None, None]

    def inverse(self, values):
        return values * self.std[:, None, None] + self.mean[:, None, None]

    def save(self, path):
        payload = asdict(self)
        payload["mean"] = self.mean.tolist()
        payload["std"] = self.std.tolist()
        payload["fit_split"] = "train"
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.pop("fit_split", None) != "train":
            raise ValueError("Normalizer must be fitted on training data only")
        payload["mean"] = np.asarray(payload["mean"], dtype=np.float32)
        payload["std"] = np.asarray(payload["std"], dtype=np.float32)
        if not np.isfinite(payload["mean"]).all() or not np.isfinite(payload["std"]).all() or np.any(payload["std"] <= 0):
            raise ValueError("Normalizer parameters must be finite with positive std")
        return cls(**payload)


def fit_channel_normalizer(archive: ResidualArchive, *, source: str = "residual",
                           chunk_size: int = 1, min_std: float = 1e-8,
                           budget_gib: float = MAX_HOST_GIB) -> ChannelNormalizer:
    """Welford combination over train samples and spatial cells, per channel."""
    if source not in {"residual", "states"} or chunk_size < 1 or min_std <= 0:
        raise ValueError("Invalid source, chunk size or minimum std")
    values = getattr(archive, source)
    start, stop = archive.splits["train"]
    count = 0
    mean = np.zeros(values.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for i in range(start, stop, chunk_size):
        check_process_memory(budget_gib)
        block = np.asarray(values[i:min(i+chunk_size, stop)], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Training normalization data contains non-finite values")
        bn = block.shape[0] * block.shape[2] * block.shape[3]
        bm = block.mean(axis=(0, 2, 3))
        bv = block.var(axis=(0, 2, 3)) * bn
        delta = bm - mean
        total = count + bn
        mean += delta * bn / total
        m2 += bv + delta ** 2 * count * bn / total
        count = total
    std = np.maximum(np.sqrt(m2 / count), min_std)
    return ChannelNormalizer(mean.astype(np.float32), std.astype(np.float32), count, source, start, stop)


def iter_era5_records(paths, variable_specs: Sequence[Mapping], *,
                      time_name: str = "time", latitude_name: str = "latitude",
                      longitude_name: str = "longitude", level_name: str = "level"):
    """Read one ERA5 timestamp at a time from NetCDF or Zarr stores.

    Each spec is {"name": "t850", "variable": "t", "level": 850}; level
    is optional for surface fields. No temporal interpolation or silent
    regridding occurs. Output dict contains time, fields [C,Y,X], latitudes,
    longitudes and channel_names. Coordinates must match across records.
    Input stores must be ordered in time. Convert to NeuralGCM's model grid
    and sigma representation with the official encoder before archive use.
    """
    import xarray as xr
    previous = None
    reference_lat = reference_lon = None
    cadence_ns = None
    if not variable_specs:
        raise ValueError("At least one variable spec required")
    for path in paths:
        opener = xr.open_zarr if str(path).endswith(".zarr") else xr.open_dataset
        kwargs = {"chunks": None} if opener is xr.open_zarr else {}
        with opener(path, **kwargs) as ds:
            times = np.asarray(ds[time_name].values, dtype="datetime64[ns]").reshape(-1)
            lat = np.asarray(ds[latitude_name].values)
            lon = np.asarray(ds[longitude_name].values)
            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
            elif not np.array_equal(lat, reference_lat) or not np.array_equal(lon, reference_lon):
                raise ValueError("ERA5 grids differ; explicitly regrid before combining")
            for i, time in enumerate(times):
                if np.isnat(time):
                    raise ValueError("Invalid ERA5 timestamp")
                if previous is not None:
                    delta = int((time-previous).astype("timedelta64[ns]").astype(np.int64))
                    if delta <= 0 or (cadence_ns is not None and delta != cadence_ns):
                        raise ValueError("ERA5 timestamps have duplicates, gaps or irregular cadence")
                    cadence_ns = delta
                previous = time
                fields = []
                for spec in variable_specs:
                    da = ds[spec["variable"]]
                    if time_name in da.dims:
                        da = da.isel({time_name: i})
                    if "level" in spec:
                        da = da.sel({level_name: spec["level"]})
                    da = da.transpose(latitude_name, longitude_name)
                    fields.append(np.asarray(da.values, dtype=np.float32))
                field = np.stack(fields)
                if not np.isfinite(field).all():
                    raise ValueError(f"Non-finite ERA5 field at {time}")
                yield {"time": time, "fields": field, "latitudes": lat,
                       "longitudes": lon, "channel_names": [s["name"] for s in variable_specs]}


def export_forecast_netcdf(output, fields, lead_hours, channel_names, latitudes,
                            longitudes, *, initialization_time, attrs=None):
    """Write one forecast [L,C,Y,X] with xarray, leaving case/member streaming to caller."""
    import xarray as xr
    expected = (len(lead_hours), len(channel_names), len(latitudes), len(longitudes))
    if fields.shape != expected:
        raise ValueError(f"Forecast shape {fields.shape} != {expected}")
    ds = xr.Dataset({"forecast": (("lead", "channel", "latitude", "longitude"), fields)},
                    coords={"lead": np.asarray(lead_hours, dtype="timedelta64[h]"),
                            "channel": list(channel_names), "latitude": latitudes,
                            "longitude": longitudes,
                            "initialization_time": np.datetime64(initialization_time)},
                    attrs=dict(attrs or {}))
    ds.to_netcdf(output)
    ds.close()
