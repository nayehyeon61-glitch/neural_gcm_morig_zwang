"""Fetch a short ERA5 period from the public ARCO store, regridded for a checkpoint.

Small-test helper: downloads the NeuralGCM inputs and forcings on all 37
pressure levels at 0.25 deg, one 6-hourly record at a time, conservatively
regrids them to the checkpoint's data grid with the same dinosaur calls as
`ERA5Reader(regrid=True)`, and writes one netCDF file per day. Point the
workflow configs at the output with `"regrid": false`.

Only `curl` is used for transfer (it follows the environment's proxy
settings); chunks are decoded with numcodecs.

Usage: python experiments/fetch_arco_era5.py --checkpoint deterministic_2_8_deg.pkl \
           --start 2019-10-01 --stop 2020-03-16 --output data/era5_tl63
"""
import argparse
import json
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import xarray as xr

STORE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
EPOCH = np.datetime64("1900-01-01T00", "h")


def fetch(name, hour=None):
    """One time slice of a variable, or the whole of a coordinate when hour is None."""
    meta = json.loads(subprocess.run(["curl", "-sSf", f"{STORE}/{name}/.zarray"], check=True,
                                     capture_output=True).stdout)
    key = ".".join([str(hour or 0)] + ["0"] * (len(meta["shape"]) - 1))
    raw = subprocess.run(["curl", "-sSf", "--retry", "4", f"{STORE}/{name}/{key}"], check=True,
                         capture_output=True).stdout
    codec = numcodecs.get_codec(meta["compressor"])
    values = np.frombuffer(codec.decode(raw), dtype=meta["dtype"]).reshape(meta["chunks"])
    return values if hour is None else values[0]


def main():
    import neuralgcm
    from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start", required=True, help="first day, inclusive")
    parser.add_argument("--stop", required=True, help="last day, exclusive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with open(args.checkpoint, "rb") as stream:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(stream))
    names = model.input_variables + model.forcing_variables
    level_hpa, latitude, longitude = fetch("level"), fetch("latitude"), fetch("longitude")
    wanted = np.asarray(model.data_coords.vertical.centers)
    levels = np.searchsorted(level_hpa, wanted)
    if not np.array_equal(level_hpa[np.minimum(levels, len(level_hpa) - 1)], wanted):
        raise ValueError("ARCO levels do not cover the checkpoint levels")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    regridder = None
    days = np.arange(np.datetime64(args.start, "D"), np.datetime64(args.stop, "D"))
    with ThreadPoolExecutor(args.workers) as pool:
        for day in days:
            path = output / f"era5_{day}.nc"
            if path.exists():
                continue
            records = []
            for hour in range(0, 24, 6):
                time = np.datetime64(day, "h") + np.timedelta64(hour, "h")
                index = int((time - EPOCH) / np.timedelta64(1, "h"))
                fields = dict(zip(names, pool.map(lambda n: fetch(n, index), names)))
                ds = xr.Dataset({n: (("level", "latitude", "longitude"), v[levels]) if v.ndim == 3
                                 else (("latitude", "longitude"), v) for n, v in fields.items()},
                                coords={"level": level_hpa[levels], "latitude": latitude,
                                        "longitude": longitude}).sortby("latitude")
                if regridder is None:
                    source = spherical_harmonic.Grid(
                        latitude_nodes=ds.sizes["latitude"], longitude_nodes=ds.sizes["longitude"],
                        latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
                        longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude))
                    regridder = horizontal_interpolation.ConservativeRegridder(
                        source, model.data_coords.horizontal, skipna=True)
                ds = xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(ds, regridder))
                records.append(ds.astype(np.float32).expand_dims(time=[time.astype("datetime64[ns]")]))
            xr.concat(records, "time").to_netcdf(str(path) + ".part", format="NETCDF4")
            Path(str(path) + ".part").rename(path)
            print(f"wrote {path.name}", flush=True)


if __name__ == "__main__":
    main()
