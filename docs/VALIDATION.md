# Verification status — 2026-10-21

This repository contains an operational research implementation, not a trained climate forecasting result.

## Verified

- Default CPU suite: **31 passed, 2 skipped**. The two expensive real-NeuralGCM tests require `RUN_NEURALGCM_INTEGRATION=1` and are run separately.
- Numerical DCT round-trip/Parseval and comparison with SciPy; known-mode separation.
- DDPM gradients, active-frequency support, seeded reproducibility and distinct samples.
- History and past-correction sensitivity; first physical interval exactly base; bounded memory over 160 six-hour-equivalent software steps.
- Chronological windows, train-only normalization, no future target in conditioning, safe NPZ checkpoint round-trip.
- Actual optimizer updates for both closure and DDPM in the synthetic CLI smoke. This is not ERA5 learning evidence.
- Metric identities, matched baseline ensembles, area weights, short-lead gate and forty-day lead requirement.
- Reader aliases, scalar time coordinates, sigma/pressure array layouts, forecast field reads stopping at issue time, exact backbone SHA checking, declared evaluation time bounds, train-only climatology.
- Actual official NeuralGCM mini-checkpoint: zero injected tendency matches baseline; positive K/s tendency warms the internal state; invalid channels/shapes/time intervals fail.
- Actual ERA5-demo workflow: native sigma residual preparation from repeated sample observations and a one-day matched forecast; disabled correction gives bitwise-equal baseline outputs. Boundary persistence also passes a deliberately narrow forcing timestamp tolerance.
- Actual enabled workflow: the history network → DDPM → time-IDCT → native sigma conversion → internal IMEX forcing path completed one day with finite output distinct from the matched baseline. Baseline output remained bitwise unchanged. The final workflow test passed in 208.49 seconds using explicitly untrained smoke-test weights.
- Dependency consistency: `pip check` passed. Exact installed package versions are recorded in `requirements-tested.txt`.

## Not established

- No real multi-year ERA5 training was performed in this workspace. The user's research-server data/checkpoint were not available here.
- No trained 40-day ERA5 hindcast, skill improvement percentage, short-term protection guarantee, or physical conservation/stability claim is reported.
- The 160-step unit test is a small synthetic controller test. The actual NeuralGCM workflow smoke is one day with repeated sample observations; neither is a substitute for real 40-day verification.
- No GPU/CUDA validation or measured production-resolution peak RAM was performed. Direct RSS/preflight checks are advisory; `scripts/run_96gb.sh` requests a genuine kernel cgroup cap and requires a working user systemd session.
- No exact MZ orthogonal dynamics, fluctuation-dissipation relationship, or Hasselmann red-spectrum reproduction has been proven.

## Real evaluation procedure

1. Prepare genuine regular ERA5 observations on the supported checkpoint grid, preserving chronological train/validation/test partitions.
2. Fit the history closure and diffusion on training residuals; choose weights by validation loss.
3. Run matched baseline/full/memory/off ensembles on many **validation** initial dates, with the whole forecast history and 40-day horizon inside that split. Select correction strength using long-lead scores subject to the day 1/3/5 regression criterion. Fall back to strength zero if none qualifies.
4. Freeze all choices, rerun on held-out **test** dates, and report per-variable RMSE/ACC/CRPS/spread with uncertainty across initial dates. Use training-derived seasonal climatology for seasonal ACC analysis.
5. Audit temporal spectra, lag correlations, ensemble calibration, extremes, moisture/energy budgets, and stability. The implemented day-wise metrics do not replace those scientific diagnostics.
