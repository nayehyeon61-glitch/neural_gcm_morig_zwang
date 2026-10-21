"""Lead-wise, area-weighted verification against the same NeuralGCM baseline.

RMSE is computed on the ensemble mean, as in NeuralGCM's reference metrics.
ACC uses explicitly supplied training-period climatology. CRPS is the exact
score of the finite empirical ensemble (not the paired training estimator).
Metrics retain individual channels; different physical units are never summed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .resources import MAX_HOST_GIB, check_process_memory


def spatial_weights(latitudes, n_longitudes: int, latitude_weights=None):
    lat = np.asarray(latitudes, dtype=np.float64)
    if lat.ndim != 1 or len(lat) < 1 or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90) or n_longitudes < 1:
        raise ValueError("Finite one-dimensional latitude degrees and positive longitude count required")
    # Supply Gaussian quadrature weights when evaluating on a Gaussian grid.
    weights = np.maximum(np.cos(np.deg2rad(lat)), 0) if latitude_weights is None else np.asarray(latitude_weights, dtype=np.float64)
    if weights.shape != lat.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Latitude weights must be finite, nonnegative and sum to a positive value")
    return weights[:, None] / (weights.sum() * n_longitudes)


def _field(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("Fields require finite [...,latitude,longitude] data")
    return values


def area_weighted_mean(values, latitudes, latitude_weights=None):
    values = _field(values)
    weights = spatial_weights(latitudes, values.shape[-1], latitude_weights)
    if values.shape[-2] != weights.shape[0]:
        raise ValueError("Latitude coordinate does not match field")
    return np.sum(values * weights, axis=(-2, -1))


def area_weighted_rmse(prediction, target, latitudes, latitude_weights=None):
    prediction, target = _field(prediction), _field(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    return np.sqrt(area_weighted_mean((prediction-target)**2, latitudes, latitude_weights))


def area_weighted_acc(prediction, target, climatology, latitudes, latitude_weights=None):
    prediction, target, climatology = _field(prediction), _field(target), _field(climatology)
    if prediction.shape != target.shape or climatology.shape != target.shape:
        raise ValueError("Prediction, target and climatology shapes must match")
    a, b = prediction-climatology, target-climatology
    numerator = area_weighted_mean(a*b, latitudes, latitude_weights)
    denominator = np.sqrt(area_weighted_mean(a*a, latitudes, latitude_weights) *
                          area_weighted_mean(b*b, latitudes, latitude_weights))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def area_weighted_crps(ensemble, target, latitudes, latitude_weights=None):
    """Empirical CRPS: mean|X-y| - (1/2)mean|X-X'|, including diagonal pairs."""
    ensemble, target = _field(ensemble), _field(target)
    if ensemble.ndim != target.ndim + 1 or ensemble.shape[1:] != target.shape or ensemble.shape[0] < 1:
        raise ValueError("Expected ensemble [member,...,latitude,longitude]")
    m = len(ensemble)
    skill = np.mean(np.abs(ensemble-target), axis=0)
    pair = np.zeros_like(target)
    for i in range(m):
        for j in range(i):
            pair += np.abs(ensemble[i]-ensemble[j])
    return area_weighted_mean(skill - pair/(m*m), latitudes, latitude_weights)


def short_range_gate(baseline_rmse, candidate_rmse, lead_days, *, tolerance: float = 0.02,
                     required_days=(1, 3, 5)):
    """Descriptive non-regression gate, not a statistical non-inferiority test.

    Every required lead and channel must satisfy candidate <= baseline*(1+tol).
    Use validation scores to select checkpoints/strengths; test scores only
    report the locked model. An absent lead raises instead of passing silently.
    """
    baseline = np.asarray(baseline_rmse, dtype=np.float64)
    candidate = np.asarray(candidate_rmse, dtype=np.float64)
    days = np.asarray(lead_days, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[0] != len(days):
        raise ValueError("RMSE arrays must have equal [lead,channel] shape")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all() or np.any(baseline < 0) or np.any(candidate < 0) or tolerance < 0:
        raise ValueError("Finite nonnegative RMSEs and tolerance required")
    rows = {}
    for day in required_days:
        indices = np.flatnonzero(np.isclose(days, day, rtol=0, atol=1e-8))
        if len(indices) != 1:
            raise ValueError(f"Exactly one lead at day {day} required for short-range gate")
        k = indices[0]
        passed = candidate[k] <= baseline[k]*(1+tolerance)
        ratio = np.divide(candidate[k], baseline[k], out=np.full_like(candidate[k], np.inf), where=baseline[k] > 0)
        ratio[(baseline[k] == 0) & (candidate[k] == 0)] = 1.0
        rows[str(day)] = {"passed_per_channel": passed.tolist(),
                          "rmse_ratio": [float(x) if np.isfinite(x) else None for x in ratio]}
    return {"passed": all(all(row["passed_per_channel"]) for row in rows.values()),
            "relative_tolerance": tolerance, "required_days": list(required_days),
            "interpretation": "descriptive validation gate; no confidence interval or skill guarantee",
            "leads": rows}


def _array(source):
    return np.load(source, mmap_mode="r", allow_pickle=False) if isinstance(source, (str, Path)) else source


def _json_values(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def evaluate_forecast_files(baseline, ensemble, truth, climatology, *, lead_days,
                            channel_names, latitudes, latitude_weights=None,
                            short_tolerance: float = 0.02, budget_gib: float = MAX_HOST_GIB,
                            evaluation_split: str = "test", climatology_provenance: str,
                            require_40_days: bool = True):
    """Stream cases, leads and members from .npy files or array-like objects.

    Shapes: truth [I,L,C,Y,X], ensemble [I,M,L,C,Y,X]. Baseline may be
    deterministic [I,L,C,Y,X] or a matched ensemble [I,Mb,L,C,Y,X]. Climatology
    must be an explicit training-derived [C,Y,X] or [I,L,C,Y,X] array, the
    latter supporting seasonal/valid-time reference values. Case/member
    trajectories are never loaded together. Pairwise CRPS uses O(M^2) reads
    but only O(C*Y*X) memory. The caller must align valid times and grids.
    """
    base, ens, obs, clim = map(_array, (baseline, ensemble, truth, climatology))
    if len(obs.shape) != 5:
        raise ValueError("Truth must be [case,lead,channel,lat,lon]")
    if not ((len(base.shape) == 5 and base.shape == obs.shape) or
            (len(base.shape) == 6 and base.shape[0] == obs.shape[0] and
             base.shape[2:] == obs.shape[1:] and base.shape[1] >= 1)):
        raise ValueError("Baseline must be [I,L,C,Y,X] or [I,Mb,L,C,Y,X] on the same cases and grid")
    ni, nl, nc, ny, nx = obs.shape
    if len(ens.shape) != 6 or ens.shape[0] != ni or ens.shape[2:] != obs.shape[1:] or ens.shape[1] < 1:
        raise ValueError("Ensemble must be [case,member,lead,channel,lat,lon]")
    if clim.shape not in {(nc, ny, nx), obs.shape}:
        raise ValueError("Climatology must be [C,Y,X] or [I,L,C,Y,X]")
    days = np.asarray(lead_days, dtype=np.float64)
    if days.shape != (nl,) or np.any(np.diff(days) <= 0) or np.any(days <= 0):
        raise ValueError("Positive strictly increasing lead days required")
    if ni < 1 or len(channel_names) != nc or len(latitudes) != ny:
        raise ValueError("Evaluation dimensions and coordinates differ")
    if evaluation_split not in {"validation", "test"} or not climatology_provenance.strip():
        raise ValueError("Declare evaluation split and training-only climatology provenance")
    if require_40_days and not np.any(np.isclose(days, 40, rtol=0, atol=1e-8)):
        raise ValueError("A 40-day lead is required for this evaluation")
    m = ens.shape[1]
    mb = base.shape[1] if base.ndim == 6 else 1
    total = {key: np.zeros((nl, nc), dtype=np.float64) for key in
             ("base_mse", "ens_mse", "base_dot", "ens_dot", "base_norm", "ens_norm", "obs_norm", "crps", "base_crps", "variance", "base_variance")}

    def member_statistics(source, i, lead, count, y):
        get = (lambda k: _field(source[i, k, lead])) if source.ndim == 6 else (lambda k: _field(source[i, lead]))
        mean, m2, absolute, pairs = (np.zeros_like(y) for _ in range(4))
        for member in range(count):
            x = get(member)
            delta = x-mean
            mean += delta/(member+1)
            m2 += delta*(x-mean)
            absolute += np.abs(x-y)
            for other in range(member):
                pairs += np.abs(x-get(other))
        variance = m2/(count-1) if count > 1 else np.zeros_like(y)
        return mean, variance, absolute/count-pairs/(count*count)

    for i in range(ni):
        for lead in range(nl):
            check_process_memory(budget_gib)
            y = _field(obs[i, lead])
            c = _field(clim if clim.ndim == 3 else clim[i, lead])
            b, base_variance, base_crps = member_statistics(base, i, lead, mb, y)
            mean, variance, crps = member_statistics(ens, i, lead, m, y)
            avg = lambda field: area_weighted_mean(field, latitudes, latitude_weights)
            a_base, a_ens, a_obs = b-c, mean-c, y-c
            total["base_mse"][lead] += avg((b-y)**2)
            total["ens_mse"][lead] += avg((mean-y)**2)
            total["base_dot"][lead] += avg(a_base*a_obs)
            total["ens_dot"][lead] += avg(a_ens*a_obs)
            total["base_norm"][lead] += avg(a_base*a_base)
            total["ens_norm"][lead] += avg(a_ens*a_ens)
            total["obs_norm"][lead] += avg(a_obs*a_obs)
            total["crps"][lead] += avg(crps)
            total["base_crps"][lead] += avg(base_crps)
            total["variance"][lead] += avg(variance)
            total["base_variance"][lead] += avg(base_variance)
    base_rmse, ens_rmse = np.sqrt(total["base_mse"]/ni), np.sqrt(total["ens_mse"]/ni)
    spread = np.sqrt(total["variance"]/ni) if m > 1 else np.full((nl, nc), np.nan)
    base_spread = np.sqrt(total["base_variance"]/ni) if mb > 1 else np.full((nl, nc), np.nan)
    def acc(prefix):
        denom = np.sqrt(total[prefix+"_norm"]*total["obs_norm"])
        return np.divide(total[prefix+"_dot"], denom, out=np.full_like(denom, np.nan), where=denom > 0)
    base_acc, ens_acc = acc("base"), acc("ens")
    spread_skill = np.divide(spread, ens_rmse, out=np.full_like(spread, np.nan), where=ens_rmse > 0)
    improvement = np.divide(base_rmse-ens_rmse, base_rmse, out=np.full_like(base_rmse, np.nan), where=base_rmse > 0)*100
    result = {"schema_version": 1, "evaluation_split": evaluation_split,
              "cases": ni, "members": m, "baseline_members": mb, "channel_names": list(channel_names),
              "climatology_provenance": climatology_provenance,
              "area_weighting": "provided latitude quadrature" if latitude_weights is not None else "cos(latitude), uniform longitude",
              "acc_aggregation": "pooled anomaly dot products over cases and spatial cells",
              "spread_estimator": "sample variance across members (ddof=1)",
              "crps_estimator": "empirical finite ensemble including diagonal pairs",
              "leads": {}}
    for k, day in enumerate(days):
        result["leads"][f"{day:g}"] = {"baseline_rmse": _json_values(base_rmse[k]),
            "ensemble_mean_rmse": _json_values(ens_rmse[k]), "rmse_improvement_percent": _json_values(improvement[k]),
            "baseline_acc": _json_values(base_acc[k]), "ensemble_mean_acc": _json_values(ens_acc[k]),
            "baseline_crps": _json_values(total["base_crps"][k]/ni), "ensemble_crps": _json_values(total["crps"][k]/ni),
            "baseline_spread": _json_values(base_spread[k]),
            "ensemble_spread": _json_values(spread[k]), "spread_skill_ratio": _json_values(spread_skill[k])}
    result["short_range_gate"] = short_range_gate(base_rmse, ens_rmse, days, tolerance=short_tolerance)
    result["long_range_checkpoints"] = {str(day): result["leads"].get(str(day)) for day in (10, 20, 30, 40)}
    return result
