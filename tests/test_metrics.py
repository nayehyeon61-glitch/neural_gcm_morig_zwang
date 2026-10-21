import numpy as np
import pytest

from neuralgcm_mz.metrics import (area_weighted_acc, area_weighted_crps,
                                  area_weighted_mean, area_weighted_rmse,
                                  evaluate_forecast_files, short_range_gate)


def test_area_weights_and_rmse():
    field = np.array([[2., 2.], [0., 0.], [2., 2.]])
    # Weights 0.5,1,0.5 -> weighted mean 1 and MSE 2.
    assert area_weighted_mean(field, [-60, 0, 60]) == pytest.approx(1)
    assert area_weighted_rmse(field, np.zeros_like(field), [-60, 0, 60]) == pytest.approx(np.sqrt(2))


def test_acc_and_exact_empirical_crps():
    y = np.array([[1., 2.], [3., 4.]])
    c = np.zeros_like(y)
    assert area_weighted_acc(2*y, y, c, [-30, 30]) == pytest.approx(1)
    assert area_weighted_acc(-y, y, c, [-30, 30]) == pytest.approx(-1)
    assert np.isnan(area_weighted_acc(c, y, c, [-30, 30]))
    ensemble = np.stack([y-1, y+1])
    # mean abs error=1; E|X-X'|=1 => CRPS=0.5.
    assert area_weighted_crps(ensemble, y, [-30, 30]) == pytest.approx(0.5)
    assert area_weighted_crps(ensemble[:1], y, [-30, 30]) == pytest.approx(1)


def test_streamed_evaluation_and_gate(tmp_path):
    days = [1, 3, 5, 10, 20, 30, 40]
    shape = (2, len(days), 2, 3, 4)
    truth = np.ones(shape)
    baseline = truth+2
    ensemble = np.stack([truth, truth+2], axis=1)
    paths = []
    for name, arr in [("baseline", baseline), ("ensemble", ensemble), ("truth", truth),
                      ("climatology", np.zeros(shape[2:]))]:
        path = tmp_path / f"{name}.npy"
        np.save(path, arr)
        paths.append(path)
    result = evaluate_forecast_files(*paths, lead_days=days, channel_names=["t", "q"],
                                     latitudes=[-60, 0, 60], climatology_provenance="training 1979-1999")
    lead = result["leads"]["40"]
    np.testing.assert_allclose(lead["baseline_rmse"], [2, 2])
    np.testing.assert_allclose(lead["ensemble_mean_rmse"], [1, 1])
    np.testing.assert_allclose(lead["ensemble_crps"], [0.5, 0.5])
    np.testing.assert_allclose(lead["ensemble_spread"], [np.sqrt(2)]*2)
    assert result["short_range_gate"]["passed"]
    assert set(result["long_range_checkpoints"]) == {"10", "20", "30", "40"}


def test_gate_requires_all_short_leads_and_channels():
    b = np.ones((3, 2))
    c = b.copy()
    c[0, 1] = 1.03
    assert not short_range_gate(b, c, [1, 3, 5])["passed"]
    with pytest.raises(ValueError, match="day 5"):
        short_range_gate(b, b, [1, 3, 6])
    zero = np.zeros_like(b)
    assert short_range_gate(zero, zero, [1, 3, 5])["passed"]
    assert not short_range_gate(zero, b, [1, 3, 5])["passed"]


def test_matched_ensemble_baseline_is_scored_as_ensemble():
    days = [1, 3, 5, 40]
    truth = np.ones((1, 4, 1, 2, 2))
    ensemble = np.stack([truth-1, truth+3], axis=1)
    result = evaluate_forecast_files(ensemble, ensemble, truth, np.zeros((1, 2, 2)),
                                    lead_days=days, channel_names=["t"], latitudes=[-30, 30],
                                    climatology_provenance="train-only")
    assert result["baseline_members"] == 2
    lead = result["leads"]["40"]
    assert lead["baseline_rmse"] == pytest.approx(lead["ensemble_mean_rmse"])
    assert lead["baseline_crps"] == pytest.approx(lead["ensemble_crps"])
    assert lead["baseline_crps"] == pytest.approx([1])
    assert lead["baseline_spread"] == pytest.approx(lead["ensemble_spread"])


def test_invalid_shapes_or_nonfinite_rejected():
    with pytest.raises(ValueError, match="shapes"):
        area_weighted_rmse(np.ones((2, 3)), np.ones((2, 2)), [-30, 30])
    with pytest.raises(ValueError, match="finite"):
        area_weighted_mean(np.array([[np.nan]]), [0])
