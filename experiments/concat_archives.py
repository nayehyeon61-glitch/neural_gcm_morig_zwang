"""Join chronologically consecutive residual archives into one, with new splits.

Each archive must continue the previous one at exactly one interval, and
share channels, grid, units and the NeuralGCM provenance (checkpoint hash,
encoder seed, memory mode). Splits are given as timestamps: train up to the
validation start, validation up to the test start, test to the end.

Usage: python experiments/concat_archives.py --inputs runs/oct_nov runs/dec_jan \
           --output runs/oct_jan --validation-start 2020-01-01T00 --test-start 2020-01-20T00
"""
import argparse
import json
from pathlib import Path

import numpy as np

from neuralgcm_mz.data import ResidualArchive

SAME_PROVENANCE = ("neuralgcm_sha256", "encoder_seed", "memory", "coordinate_system", "variables")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--test-start", required=True)
    args = parser.parse_args()
    parts = [ResidualArchive(p) for p in args.inputs]
    first = parts[0]
    for previous, part in zip(parts, parts[1:]):
        if part.manifest["channel_names"] != first.manifest["channel_names"] or part.states.shape[1:] != first.states.shape[1:]:
            raise ValueError("Archives differ in channels or grid")
        if part.dt_seconds != first.dt_seconds or not np.array_equal(part.latitudes, first.latitudes):
            raise ValueError("Archives differ in cadence or latitudes")
        for key in SAME_PROVENANCE:
            if part.manifest["provenance"].get(key) != first.manifest["provenance"].get(key):
                raise ValueError(f"Archives differ in provenance {key}")
        step = np.timedelta64(int(first.dt_seconds), "s")
        if part.timestamps[0] != previous.timestamps[-1] + step:
            raise ValueError("Archives are not consecutive")
    times = np.concatenate([p.timestamps for p in parts])
    n = len(times)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shape = (n, *first.states.shape[1:])
    for name in ("states", "residual"):
        out = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype="float32", shape=shape)
        start = 0
        for part in parts:
            source = getattr(part, name)
            for i in range(len(source)):
                out[start + i] = source[i]
            start += len(source)
        out.flush()
    np.save(output / "timestamps.npy", times)
    np.save(output / "latitudes.npy", first.latitudes)
    index = lambda t: int(np.searchsorted(times, np.datetime64(t, "ns")))
    validation, test = index(args.validation_start), index(args.test_start)
    manifest = dict(first.manifest)
    manifest["shape"] = list(shape)
    manifest["splits"] = {"train": [0, validation], "validation": [validation, test], "test": [test, n]}
    manifest["provenance"] = {**first.manifest["provenance"], "joined_from": [str(p) for p in args.inputs]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ResidualArchive(output)  # validates the result
    print(json.dumps({"intervals": n, "splits": manifest["splits"]}))


if __name__ == "__main__":
    main()
