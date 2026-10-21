import json
from types import SimpleNamespace

import numpy as np
import pytest

from neuralgcm_mz.training import (
    TrainingConfig, load_checkpoint, normalize_history, sample_batch,
    save_checkpoint, train,
)


def test_safe_checkpoint_roundtrip(tmp_path):
    payload = {"params": {"w": np.arange(6, dtype=np.float32).reshape(2, 3)},
               "sequence": (np.array([3.0]), [1, "a", None]), "config": {"steps": 8}}
    path = tmp_path / "weights.npz"
    save_checkpoint(path, payload)
    result = load_checkpoint(path)
    np.testing.assert_array_equal(result["params"]["w"], payload["params"]["w"])
    assert isinstance(result["sequence"], tuple)
    assert result["sequence"][1] == [1, "a", None]
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    with pytest.raises(TypeError):
        save_checkpoint(tmp_path / "bad.npz", {"bad": np.array([object()], dtype=object)})


def test_training_config_rejects_inconsistent_settings(tmp_path):
    with pytest.raises(ValueError, match="96"):
        TrainingConfig(memory_limit_gb=97)
    with pytest.raises(ValueError, match="DCT"):
        TrainingConfig(max_period_hours=12)
    config = tmp_path / "wrong.json"
    config.write_text(json.dumps({"batch_size_typo": 3}))
    with pytest.raises(ValueError, match="Unknown"):
        TrainingConfig.from_json(config)


def test_history_contains_only_past_residuals():
    shape = (24, 1, 2, 2)
    time = np.arange(shape[0], dtype=np.float32)[:, None, None, None]
    archive = SimpleNamespace(states=np.broadcast_to(time, shape),
                              residual=np.broadcast_to(1000 + time, shape),
                              latitudes=np.array([-30, 30]))
    norm = {"state_mean": np.array([0.]), "state_std": np.array([1.]),
            "residual_mean": np.array([0.]), "residual_std": np.array([1.])}
    config = TrainingConfig(history_steps=4, block_steps=4, batch_columns=2)
    history, target = sample_batch(archive, [8], config, norm, np.random.default_rng(0))
    np.testing.assert_array_equal(history[0, :, 0], [5, 6, 7, 8])
    np.testing.assert_array_equal(history[0, :, 1], [1004, 1005, 1006, 1007])
    np.testing.assert_array_equal(target[0, :, 0], [1008, 1009, 1010, 1011])
    assert history.shape == (2, 4, 2)
    with pytest.raises(ValueError, match="matching"):
        normalize_history(np.ones((2, 4, 1)), np.ones((2, 3, 1)), norm)


def test_train_uses_train_normalizer_and_persists_both_models(tmp_path):
    from neuralgcm_mz.data import write_residual_archive
    n = 64
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(n + 1, 2, 2, 3)).astype(np.float32)
    # A conspicuous held-out offset must not contaminate fitted statistics.
    truth[40:] += 100
    tendency = rng.normal(0, 1e-4, size=(n, 2, 2, 3)).astype(np.float32)
    baseline = truth[1:] - 21600 * tendency
    timestamps = np.datetime64("2000-01-01") + np.arange(n + 1) * np.timedelta64(6, "h")
    write_residual_archive(truth, baseline, timestamps, tmp_path / "archive",
                           channel_names=["T", "u"], channel_units=["K", "m s^-1"],
                           latitudes=[-45, 45],
                           splits={"train": [0, 40], "validation": [40, 52], "test": [52, 64]})
    config = TrainingConfig(history_steps=3, block_steps=3, hidden=8,
                            diffusion_steps=4, batch_columns=2,
                            closure_updates=2, diffusion_updates=2,
                            validation_every=1, validation_batches=1,
                            normalization_chunk_steps=2)
    path = tmp_path / "model.npz"
    summary = train(tmp_path / "archive", path, config)
    model = load_checkpoint(path)
    assert not summary["test_used"]
    assert summary["normalization_fit_split"] == "train"
    np.testing.assert_allclose(model["normalizer"]["state_mean"], truth[:40].mean(axis=(0, 2, 3)), atol=1e-6)
    assert np.isfinite(summary["diffusion_validation_loss"])
    assert model["closure_params"] and model["diffusion_params"]
    assert path.with_suffix(".training.json").exists()
