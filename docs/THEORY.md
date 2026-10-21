# Theory and scope of the research adapter

이 구현은 NeuralGCM의 기존 learned physics에 **과거 상태·과거 보정량을 사용하는 메모리 근사**와 **시간 DCT 고주파 성분을 생성하는 DDPM**을 더하는 연구용 adapter입니다. 첫 예측 구간은 기본 NeuralGCM만 사용합니다. 이후 40일까지 반복 적분하지만, 장기 예측력 개선이나 단기 예측력 보존이 입증되었다는 의미는 아닙니다.

## What is actually modeled

The frozen NeuralGCM provides the dynamical core and the original learned physics. The added closure has two trainable components:

1. A finite-history network predicts the conditional mean of a future residual-tendency block.
2. A conditional denoising diffusion model generates the fast temporal part of the remaining innovation.

Both are shared column-wise, all-channel MLPs. A column contains the selected variables and vertical levels; weights are shared across horizontal locations. These networks do not implement a horizontal U-Net, attention over the global grid, or separately supervised physical-process experts. Horizontal propagation remains in NeuralGCM's dynamics. Independent column draws do not by themselves reproduce the spatial covariance of unresolved weather; evaluate spatial spectra and cross-column correlations before interpreting ensemble structure.

For a batch of columns, write the causal context as

\[
 h_n = \operatorname{concat}\bigl(
   \operatorname{norm}(x_{n-H+1:n}),
   \operatorname{norm}(a_{n-H:n-1})
 \bigr) \in \mathbb R^{B\times H\times 2C}.
\]

Here, each state is paired with the preceding interval's correction. During archive training, previous corrections are teacher-forced residual-tendency targets. During free rollout, they are previous realized model corrections; future verification states are unavailable to the closure. This training/rollout difference is a source of distribution shift.

The history network predicts

\[
 \mu_\theta(h_n)\in\mathbb R^{B\times K\times C}.
\]

It is trained against the **full normalized residual block**, before frequency filtering. Consequently, the deterministic conditional mean can represent slow bias and history-dependent responses. The same network also sees previous realized fast corrections, allowing past stochastic forcing to influence later mean predictions. This is a finite-memory approximation, not an explicit exact Mori–Zwanzig convolution kernel.

## Residual target and coordinate contract

In a common state coordinate system, the offline target is

\[
 r_n = \frac{x^{\mathrm{target}}_{n+1}
              -\Phi_{\mathrm{base},\Delta t}(x^{\mathrm{target}}_n,c_n)}
             {\Delta t}.
\]

The forecast is teacher-forced: each baseline interval starts from the corresponding target state. A long free-running forecast error contains accumulated trajectory error and is not interchangeable with this one-step target.

For internal NeuralGCM correction, both terms must be expressed on its **sigma-level nodal model grid**, with identical variables, vertical levels, units and ordering. When ERA5 is mapped through the frozen encoder to form sigma targets, the target is a discrepancy in the encoded model representation. It includes encoder/regridding error and representation mismatch. Encoding observed endpoints separately does not reveal the unique true unresolved tendency; use deterministic/reproducible encoding and record the checkpoint and conversion procedure.

Dividing endpoint error by the data interval produces an **interval-averaged tendency proxy**. It is not the exact instantaneous missing physics. Injecting this constant proxy through a nonlinear solver does not guarantee exact endpoint recovery. Its quality and stability must be assessed through independent rollouts.

The backend takes temperature tendencies in K/s, wind tendencies in m/s² and tracer tendencies in 1/s on `[sigma, longitude, latitude]`. It converts physical units and horizontal vector tendencies into NeuralGCM's spectral state representation. Pressure-level residual arrays cannot be passed directly to that interface. Temperature and moisture outputs are not uniquely attributable to radiation, condensation, turbulence or other specific processes.

## Temporal DCT and DDPM objective

Let `DCT_t` denote an orthonormal DCT-II along the future block's **time axis**. Define the mean-centered innovation and its retained coefficients by

\[
 z_n = P_{\mathrm{fast}}\operatorname{DCT}_t
       \bigl(r_{n:n+K-1}-\mu_\theta(h_n)\bigr).
\]

All quantities in this expression are normalized using training-only channel statistics. The history mean is fit first; the denoiser then learns its remaining high-frequency innovation. The DDPM uses Gaussian corruption and a Gaussian starting distribution:

\[
 z_s = \sqrt{\bar\alpha_s}z_n
       +\sqrt{1-\bar\alpha_s}\epsilon,
 \qquad \epsilon\sim\mathcal N(0,I),
\]

with an epsilon-prediction objective restricted to retained modes. At inference, reverse DDPM sampling produces `z_hat`; the normalized correction block is

\[
 \widehat a_{n:n+K-1}
   = \mu_\theta(h_n)+\operatorname{IDCT}_t(\widehat z_n).
\]

It is then converted back to physical tendency units. Gaussian corruption does **not** assert that real residuals or the generated distribution are Gaussian. The algorithm is DDPM, not flow matching, and its denoising index `s` is distinct from physical time `t`.

For `K` samples separated by `Δt`, the DCT-II mode frequencies are

\[
 f_k=\frac{k}{2K\Delta t},\qquad
 T_k=\frac{2K\Delta t}{k}\quad(k>0).
\]

The DC mode has infinite period and is excluded from the fast branch. A physical cutoff retains positive modes with `T_k <= max_period_hours`. DCT coefficient index is frequency, not time, and the transform does not itself establish a physical causal direction. Causality comes from the history/target boundary and inference inputs.

Sampling cadence limits what can be learned. Daily ERA5 cannot identify subdaily fluctuations; even an arbitrarily detailed interpolation of daily endpoints does not recover them. The Nyquist period is two sampling intervals, while the largest finite-block DCT-II mode lies just below the Nyquist frequency. Configurations with no retained mode are rejected. Six-hour data can represent shorter periods than daily data, but neither resolves arbitrary fast physics.

## Integration inside learned physics

For an interval whose sampled correction is `a_n`, the intended modified dynamics are

\[
 \dot x = F_{\mathrm{core}}(x)
          +G_{\mathrm{base}}(x,c)+a_n.
\]

The backend intercepts the supported NeuralGCM `DycoreWithPhysicsCorrector` call, adds the mapped tendency to the original learned-physics tendency, and uses the unchanged upstream dynamical integration. This is an internal forcing hook rather than a change to decoded output. Unsupported integrator configurations raise an error instead of silently changing the operation.

The first forecast interval uses the unmodified baseline. Subsequent correction blocks depend on available history; generated corrections and evolving states are retained for later blocks. Each forcing realization remains fixed over its physical interval and all internal solver stages. A new random sample is not drawn on every RHS evaluation. Each ensemble member needs its own random stream and persistent history.

This is a random ordinary differential equation driven by finite-interval forcing. The physical contribution scales with `dt`. It is not a white-noise SDE and does not use a Brownian `sqrt(dt)` increment. Sampling more DDPM denoising steps changes the generative approximation, not the physical forecast duration.

The original NeuralGCM parameters remain frozen. Offline gradient updates train the closure and denoiser. This implementation does not claim end-to-end optimization through a 40-day NeuralGCM rollout, nor modification of every hidden layer in the original learned-physics network. Its learned history is applied through the added network and the internal tendency hook.

## Relationship to Mori–Zwanzig and Hasselmann

Mori–Zwanzig reduction distinguishes a resolved/Markovian term, a history term and an orthogonal-dynamics term. The attached Venturi–Li paper derives such relations for neural-network layer propagation; its layer-depth memory is not automatically atmospheric-time memory. The paper explicitly distinguishes MZ noise from externally injected network noise. Here, history-conditioned closure and stochastic innovation approximate these roles at the modeling level. No projection operator or exact orthogonal dynamics is computed, and no fluctuation–dissipation theorem is enforced.

Hasselmann's construction concerns a slow system's response to rapidly decorrelating forcing, with stabilizing feedback yielding stationary red variability. It motivates keeping past shocks relevant to subsequent dynamics. It does not imply that an individual future weather shock is known, or that adding noise improves forecast accuracy.

**Strict high-pass filtering alone has no zero-frequency forcing.** Removing the DC coefficient makes the generated fast branch's sampled sum zero over each full block, so a linear accumulation of those samples has no stochastic block-mean increment. Hasselmann forcing, by contrast, decorrelates rapidly but has nonzero spectral density near zero frequency. The forcing therefore has a second, **slow stochastic branch**.

## Zero-frequency AR(1) branch

After the history mean is fitted, the block mean of the training innovation

\[
 d_n = \frac1K\sum_{j=0}^{K-1}\bigl(r_{n+j}-\mu_\theta(h_n)_j\bigr)
\]

is exactly the part the fast DDPM cannot generate. Its channel covariance \(\Sigma=LL^\top\) and its pooled lag-one correlation between consecutive blocks \(\phi\) are fitted on training windows. During rollout each block receives a block-constant normalized forcing

\[
 s_{b+1}=\phi s_b+\sqrt{1-\phi^2}\,L z_b,\qquad s_0=Lz_0,
\]

which is stationary with covariance \(\Sigma\). For a slow variable with damping \(\lambda\), \(\dot y=-\lambda y+s\), the response variance grows like that of a random walk until \(t\sim1/\lambda\), then saturates with a red spectrum: the Hasselmann mechanism. Zero-sum fast forcing gives no such accumulation (`tests/test_long_term_noise.py` checks both). A single pooled \(\phi\) keeps the vector process stationary for any \(\Sigma\); channel-dependent memory is a possible extension.

## Spatially correlated draws

Column-wise networks with independent column noise produce spatially white forcing. On TL63 about half of its variance is beyond the spectral truncation and the resolved remainder sits at the smallest scales, where hyperdiffusion removes it (a 6-hour temperature response of 38% of the injected amount, against 86% for a smooth pattern). All Gaussian draws, the DDPM prior and every reverse-step noise as well as the AR(1) shocks, therefore come from isotropic fields with spectral variance \(\exp(-l(l+1)L^2/(2R^2))\) (`noise_correlation_km`), normalized to unit pointwise variance. Because the denoiser is column-wise, each column's marginal reverse process is unchanged; only the cross-column covariance becomes realistic. Training estimates the zonal correlation length of the fast and block-mean innovations separately (`correlation_km_fast`, `correlation_km_slow`) and forecasts use them unless overridden; 500 km is only the fallback for older checkpoints. Draws are independent across (mode, channel), so inter-channel structure comes from the denoiser alone.

## Amplitude calibration and memory consistency

A small denoiser with few reverse steps does not reproduce the target variance; a known Gaussian test showed 1.35–1.8× inflation, while the exact-score sampler itself under-disperses by 6–13%. After training, a per-(mode, channel) gain matching sample spread to held-out innovation spread is fitted on validation windows and applied to the DCT coefficients.

The history network was trained on past *residuals*. In rollout the memory therefore stores the closure's own unscaled residual estimate (mean + noise, before `correction_scale`), and only the applied forcing is scaled. The pure-base first interval stores the training-mean residual, which normalizes to zero, instead of an out-of-distribution exact zero.

The history mean, prior sampled corrections conditioning later means, and the nonlinear resolved dynamics are additional slow pathways. None of this proves a realistic low-frequency spectrum; measure rollout spectra, autocorrelation and drift, and compare `full`, `memory` (mean only), `noise` (noise only) and `off`.

## Fast-noise strength

`fast_noise_scale` multiplies only the DDPM branch; `slow_noise_scale` the AR(1) branch. Correlated draws change the short-range impact of the same nominal amplitude: on the TL63 core with a unit-variance (gain-normalized) untrained sampler, the day-1 temperature response was 0.093 K for independent columns and 0.174 K for correlated draws (day 5: 0.32 and 0.47 K), at equal forcing rms of 0.75 K/day. Keeping the previous short-range impact therefore needs a fast scale near 0.5, which the example configurations use as a conservative start. The real choice is made on validation data with `ngcm-mz-workflow calibrate` (`configs/calibrate.example.json`): every case must pass the day 1/3/5 RMSE gate, and among those the scale whose median spread/skill is closest to 1 (at most 1 + tolerance) wins, else 0.

The clip (`tendency_clip_std`) acts after the strengths. If raw closure output exceeds it, the clip and not the strength sets the amplitude; an untrained sampler produced about 300σ and made every scale identical until its gain was normalized. Check the clipped fraction on trained checkpoints.

With K = 8 six-hour steps and a 48 h cutoff, DCT mode k = 1 (96 h period) is in neither branch: it is carried only by the history mean.

## What 40-day evaluation establishes

Forty days is a subseasonal forecast horizon. It is not enough to validate climate statistics, a stationary invariant distribution, or multidecadal climate response. NeuralGCM already has published long climate simulations under prescribed surface forcing; the hypothesis here is improved subseasonal forecast distributions and memory-sensitive errors relative to the same frozen base.

Compare identical initializations, coordinates, forcing policies and verification periods. Future observed SST/sea ice used in historical experiments must be labeled as prescribed hindcast forcing, not information available to a real-time forecast. Fit climatology and normalization on the training split only; keep complete history and target windows within each chronological split.

Report lead-wise area-weighted RMSE, bias, ensemble CRPS and spread-skill alongside weekly anomaly skill, drift and temporal/spatial spectra. Separate days 1–5, 6–15 and 16–40. A single trajectory or synthetic smoke test is not evidence of ERA5 forecast improvement.

Freezing the base, starting with a pure-base interval, and restricting correction magnitude can reduce interference; they do not guarantee short-range skill preservation after the first interval. Correction strength and any deployment gate require held-out rollout validation with an explicit short-range non-inferiority threshold. Memory-only, diffusion-without-memory, full-model and cutoff ablations are needed to distinguish the mechanism of any improvement.

See [SOURCES.md](SOURCES.md) for exact source provenance and the differences between this adapter and its references.
