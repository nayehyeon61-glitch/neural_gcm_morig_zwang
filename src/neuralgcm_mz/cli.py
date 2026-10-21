"""Command line tools. Run ``ngcm-mz --help`` for concrete input contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _print_result(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def smoke(output):
    """Small synthetic training test; never represents ERA5 forecast skill."""
    import numpy as np
    from .data import write_residual_archive
    from .training import TrainingConfig, train

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    n, c, y, x = 160, 2, 3, 4
    truth = rng.normal(size=(n + 1, c, y, x)).astype(np.float32)
    correction = (0.02 * np.sin(np.arange(n)[:, None, None, None] / 3)
                  + 0.01 * truth[:-1]) / 21600
    baseline = truth[1:] - 21600 * correction
    timestamps = np.datetime64("2000-01-01T00") + np.arange(n + 1) * np.timedelta64(6, "h")
    write_residual_archive(truth, baseline, timestamps, output / "archive",
                           channel_names=["synthetic_a", "synthetic_b"],
                           channel_units=["1", "1"], latitudes=[-60, 0, 60],
                           provenance={"synthetic": True, "purpose": "software smoke test only"})
    config = TrainingConfig(history_steps=4, block_steps=4, hidden=16,
                            diffusion_steps=8, batch_columns=4,
                            closure_updates=3, diffusion_updates=3,
                            validation_every=1, validation_batches=1,
                            memory_limit_gb=8, normalization_chunk_steps=2)
    summary = train(output / "archive", output / "smoke_model.npz", config)
    return {"synthetic_only": True, "checkpoint": str(output / "smoke_model.npz"),
            "training_summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description="NeuralGCM causal history / time-DCT / diffusion closure")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="Fit closure and DDPM using a prepared residual archive")
    p.add_argument("--archive", required=True)
    p.add_argument("--config", help="TrainingConfig JSON; default is the 6-hour 96 GB configuration")
    p.add_argument("--output", required=True, help="Output safe .npz checkpoint")
    p = sub.add_parser("evaluate", help="Compare aligned pressure-level forecast .npy arrays")
    for name in ("baseline", "ensemble", "truth", "climatology", "metadata", "output"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--short-tolerance", type=float, default=0.02)
    p.add_argument("--memory-limit-gb", type=float, default=96.0)
    p = sub.add_parser("inspect", help="Print checkpoint configuration and training summary")
    p.add_argument("checkpoint")
    p = sub.add_parser("smoke", help="Synthetic CPU training/checkpoint smoke test (not a skill evaluation)")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        from .training import TrainingConfig, train
        config = TrainingConfig.from_json(args.config) if args.config else TrainingConfig()
        _print_result(train(args.archive, args.output, config))
    elif args.command == "evaluate":
        from .metrics import evaluate_forecast_files
        if not 0 < args.memory_limit_gb <= 96:
            parser.error("memory-limit-gb must be in (0,96]")
        metadata = json.loads(Path(args.metadata).read_text())
        allowed = {"lead_days", "channel_names", "latitudes", "latitude_weights", "evaluation_split", "climatology_provenance"}
        unknown = set(metadata) - allowed
        if unknown:
            parser.error(f"Unknown evaluation metadata keys: {sorted(unknown)}")
        report = evaluate_forecast_files(args.baseline, args.ensemble, args.truth, args.climatology,
                                        short_tolerance=args.short_tolerance,
                                        budget_gib=args.memory_limit_gb * 1e9 / 1024**3, **metadata)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        _print_result({"report": str(destination), "short_range_gate": report["short_range_gate"]})
    elif args.command == "inspect":
        from .training import load_checkpoint
        checkpoint = load_checkpoint(args.checkpoint)
        _print_result({key: checkpoint[key] for key in ("config", "channel_names", "state_units", "residual_units", "dt_seconds", "provenance", "training_summary")})
    elif args.command == "smoke":
        _print_result(smoke(args.output))


if __name__ == "__main__":
    main()
