# Sources and implementation provenance

제공된 네 논문과 지정된 네 저장소를 검토했습니다. 아래 commit은 검토한 소스를 고정해 추적하기 위한 것입니다. 원본의 전체 구현을 재현했다는 의미는 아니며, 잔차·시간 메모리·diffusion을 연결하는 부분은 이 저장소의 연구용 adapter입니다.

## Source checkouts inspected

| Repository | Inspected commit | Role |
|---|---|---|
| [neuralgcm/neuralgcm](https://github.com/neuralgcm/neuralgcm) | `532ec671b1372b9a8743f94d94cb3e648e07ddd0` | Frozen baseline and internal dynamical/learned-physics integration. |
| [bhouri0412/Hist_Bayesian_Closure](https://github.com/bhouri0412/Hist_Bayesian_Closure) | `0c555780dc4e9a40d404ddf422730e7907803e60` | Finite-history, per-variable closure and integration of closure in the physical RHS. |
| [gcouairon/ArchesWeather](https://github.com/gcouairon/ArchesWeather) | `17e08b1f83e228e1e57a2fe27c4b931ea7b5ade5` | Deterministic weather baseline structure and the official successor link. |
| [robbiewatt1/ClimateDiffuse](https://github.com/robbiewatt1/ClimateDiffuse) | `7b8ee181fa3baeacf08a2bf9dc275989048b4030` | Residual normalization and conditional Gaussian-corruption diffusion for climate fields. |
| [INRIA/geoarches](https://github.com/INRIA/geoarches) | `b2bdb0766749aa484c422da134e9b77827026ded` | Official successor named by ArchesWeather; inspected for ArchesWeatherGen residual targets. |

These are source-review hashes. Runtime package constraints and tested versions are recorded separately in the project dependency files; they should not be inferred from this table.

## Concrete source-to-adapter mapping

### NeuralGCM

The implementation uses the checkpoint-based `PressureLevelModel` API and a supported `DycoreWithPhysicsCorrector` hook. Relevant upstream sources are the [legacy API](https://github.com/neuralgcm/neuralgcm/blob/532ec671b1372b9a8743f94d94cb3e648e07ddd0/neuralgcm/legacy/api.py), [correctors](https://github.com/neuralgcm/neuralgcm/blob/532ec671b1372b9a8743f94d94cb3e648e07ddd0/neuralgcm/legacy/correctors.py), and [transforms](https://github.com/neuralgcm/neuralgcm/blob/532ec671b1372b9a8743f94d94cb3e648e07ddd0/neuralgcm/legacy/transforms.py).

The new code maps SI sigma-grid tendencies to the spectral model state and adds them inside the original learned-physics corrector. It does not replace the primitive equations or implement a new general circulation model. Support for a checkpoint must be checked against its integrator configuration; this is not a universal hook for every current or future NeuralGCM API.

### Hist_Bayesian_Closure

The [code guide](https://github.com/bhouri0412/Hist_Bayesian_Closure/blob/0c555780dc4e9a40d404ddf422730e7907803e60/readme.txt) identifies the history and non-history experiments. In [forward_pass.py](https://github.com/bhouri0412/Hist_Bayesian_Closure/blob/0c555780dc4e9a40d404ddf422730e7907803e60/Code/forward_pass.py), `forward_pass_2`, `_3`, `_4`, `_5` and `_10` concatenate a resolved variable's current and lagged values before applying its MLP. In [stepper.py](https://github.com/bhouri0412/Hist_Bayesian_Closure/blob/0c555780dc4e9a40d404ddf422730e7907803e60/Code/stepper.py), `L96_2t_xdot_*` adds the learned closure to the Lorenz-96 RHS, while `stepper_*` applies RK4 with history. [Hist_Deterministic.py](https://github.com/bhouri0412/Hist_Bayesian_Closure/blob/0c555780dc4e9a40d404ddf422730e7907803e60/Code/Hist_Deterministic.py) provides the `ODEfit` training reference.

This adapter retains the history-to-tendency idea but uses shared all-channel column MLPs, also conditions on past corrections, predicts a block, and uses NeuralGCM's own solver. It does not use the original Lorenz-96 equations, copy the original per-variable architecture unchanged, or reproduce its Bayesian HMC weight inference. Its uncertainty model is DDPM-generated innovation.

No standalone `morig-zwang.txt` attachment was supplied in this session. The available text guide is the repository's `readme.txt`; the supplied MZ theory source is the Venturi–Li PDF.

### ArchesWeather and its official successor

The specified [ArchesWeather README](https://github.com/gcouairon/ArchesWeather/blob/17e08b1f83e228e1e57a2fe27c4b931ea7b5ade5/README.md) marks that repository unmaintained and directs users to Geoarches. Its [forecast module](https://github.com/gcouairon/ArchesWeather/blob/17e08b1f83e228e1e57a2fe27c4b931ea7b5ade5/lightning_modules/forecast.py) is a deterministic forecaster; the requested checkout does not contain ArchesWeatherGen residual diffusion.

In the official successor's [diffusion module](https://github.com/INRIA/geoarches/blob/b2bdb0766749aa484c422da134e9b77827026ded/geoarches/lightning_modules/diffusion.py), `training_step` with `learn_residual="pred"` constructs `next_state - pred_state`, and `sample` adds the generated residual to the deterministic prediction. That inspected module uses `FlowMatchEulerDiscreteScheduler`.

This project takes the **conditional residual target principle**, replaces the deterministic forecast with NeuralGCM, converts residuals to interval-average tendency proxies, filters the temporal innovation using DCT, and explicitly uses DDPM instead of flow matching. It injects the result inside learned physics rather than simply adding a field to a decoded forecast.

### ClimateDiffuse

[DatasetUS.py](https://github.com/robbiewatt1/ClimateDiffuse/blob/7b8ee181fa3baeacf08a2bf9dc275989048b4030/src/DatasetUS.py) defines `UpscaleDataset`: its target is a normalized spatial residual, fine field minus interpolated coarse field. [TrainDiffusion.py](https://github.com/robbiewatt1/ClimateDiffuse/blob/7b8ee181fa3baeacf08a2bf9dc275989048b4030/src/TrainDiffusion.py) defines `EDMLoss` with Gaussian corruption and sigma-dependent weighting, plus `sample_model` with an Euler/Heun noise-level sampler. [Network.py](https://github.com/robbiewatt1/ClimateDiffuse/blob/7b8ee181fa3baeacf08a2bf9dc275989048b4030/src/Network.py) contains `EDMPrecond`; [Inference.py](https://github.com/robbiewatt1/ClimateDiffuse/blob/7b8ee181fa3baeacf08a2bf9dc275989048b4030/src/Inference.py) contains `sample_model_EDS`.

ClimateDiffuse is an EDM spatial downscaling implementation. This adapter is a DDPM temporal-coefficient generator with a smaller conditional MLP and does not copy its spatial U-Net, EDM preconditioning, ERA5 regional dataset loader or sampler. The two diffusion formulations are different; this is a conceptual adaptation of residual normalization and conditional denoising, not a claim to run the original ClimateDiffuse model.

## Papers supplied by the user

1. Daniele Venturi and Xiantao Li. **The Mori–Zwanzig formulation of deep learning.** [arXiv:2209.05544v4](https://arxiv.org/abs/2209.05544v4), 2023. Supplied file: `2209.05544v4 (1).pdf`. Equation (97) and its discussion separate streaming, orthogonal-dynamics noise and memory. The paper's network-layer reduction motivates the distinction between learned history and injected noise; it does not directly derive this atmospheric closure.
2. Dmitrii Kochkov et al. **Neural general circulation models for weather and climate.** Nature, 2024. [DOI:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y). Supplied file: `s41586-024-07744-y (4).pdf`. Provides the hybrid model and baseline evaluation context. NeuralGCM already supports long simulations under specified forcing; the experiment here targets a particular subseasonal improvement hypothesis.
3. Klaus Hasselmann. **Stochastic climate models. Part I. Theory.** Tellus, 1976. [DOI:10.1111/j.2153-3490.1976.tb00696.x](https://doi.org/10.1111/j.2153-3490.1976.tb00696.x). Supplied file: `Tellus - December 1976 - Hasselmann - Stochastic climate models Part I  Theory (1)(1).pdf`. Motivates short-correlation weather forcing, slow response and stabilizing feedback. Strict high-pass block forcing is an additional modeling choice and is not a direct reproduction of Hasselmann's forcing assumptions.
4. Lin Lin, Hans von Storch and Xueen Chen. **The Stochastic Climate Model helps reveal the role of memory in internal variability in the Bohai and Yellow Sea.** Communications Earth & Environment, 2023. [DOI:10.1038/s43247-023-01018-7](https://doi.org/10.1038/s43247-023-01018-7). Supplied file: `s43247-023-01018-7 (3).pdf`. Supports investigating persistence, scale dependence and response spectra. Its regional ocean simulation findings are not validation of this global atmospheric adapter.

Related source papers identified through the repositories are Bhouri and Gentine's [History-Based, Bayesian, Closure for Stochastic Parameterization: Application to Lorenz '96](https://arxiv.org/abs/2210.14488), the original [ArchesWeather paper](https://arxiv.org/abs/2405.14527), and Watt and Mansfield's [Generative Diffusion-based Downscaling for Climate](https://arxiv.org/abs/2404.17752).

## Attribution and reproducibility boundary

The reference code was inspected to establish the algorithmic mapping above; no upstream source files are vendored in this implementation. Upstream packages and checkpoints retain their respective terms and attribution. Record the actual installed NeuralGCM version, checkpoint hash, ERA5 preprocessing metadata, sample cadence, normalization split, DCT cutoff, random seeds and adapter checkpoint for each result. Source inspection and numerical unit tests do not replace training or held-out forecast evaluation.

## Tested NeuralGCM release and internal hook

The executable integration test uses **NeuralGCM 1.1.2**, Git tag commit
[`1e5e3f67f3c36cb087f432e1b368160fb98fc124`](https://github.com/neuralgcm/neuralgcm/tree/1e5e3f67f3c36cb087f432e1b368160fb98fc124).
In this release, [api.py](https://github.com/neuralgcm/neuralgcm/blob/1e5e3f67f3c36cb087f432e1b368160fb98fc124/neuralgcm/api.py),
[correctors.py](https://github.com/neuralgcm/neuralgcm/blob/1e5e3f67f3c36cb087f432e1b368160fb98fc124/neuralgcm/correctors.py), and
[transforms.py](https://github.com/neuralgcm/neuralgcm/blob/1e5e3f67f3c36cb087f432e1b368160fb98fc124/neuralgcm/transforms.py)
live directly under `neuralgcm/`; the newer checkout above relocates them into
`neuralgcm/legacy/`. The tests establish support for the pinned release, not
for the newer modern NeuralGCM API.

The hook uses Haiku `intercept_methods`, preserves the original parameter
namespaces, and adds the converted closure tendency immediately before
`DycoreWithPhysicsCorrector` composes the explicit forcing with its dycore.
For checkpoints using `CustomCoordsCorrector`, it applies Dinosaur's spectral
interpolator to the actual dycore grid before addition. SI wind tendencies
follow the upstream cosine-latitude and div/curl convention. Noise is held
fixed across integrator stages. Persistence boundary conditions relabel the
same prescribed SST/sea-ice values at the current model time; they do not
read future observations.
