# Changelog

All notable changes are recorded here ([Keep a Changelog](https://keepachangelog.com)).

## [Unreleased]

### Added
- `dynecon.estimation.hmc`: own Hamiltonian Monte Carlo sampler with dual-averaging step size, diagonal
  mass adaptation and divergence counting; `MCMCResult.divergences`.
- `smm_exact.quasi_posterior`: HMC over the Chernozhukov-Hong quasi-posterior of the moment
  conditions using the exact moment Jacobians (not a likelihood; tested on analytic moments only).
- `hmc(..., inv_mass=...)`: fixed dense or diagonal mass matrix (e.g. the inverse Hessian at a mode).
- `mixture_moments` and `patient=True` in `smm_exact` / `quasi_posterior`: a patient minority
  (`beta_patient`, `patient_share` on `consumption_savings`) with exact derivatives. Fitted to the
  2022 SCF it matches five wealth moments (just-identified); the single-type model is rejected.
- `GridWarning`: raised by `stationary_distribution_young` when more than 1% of the mass sits on the
  top asset node (a silent truncation before; patient households had mean assets biased by half).
- Scripts for real data (`scripts/real_scf_*.py`, `scripts/real_sw_*.py`): SCF wealth moments with a
  household bootstrap, and a Dynare linear-model reader that reproduces the Smets-Wouters (2007)
  posterior mode (log posterior -835.7 against Dynare's -836.1).

### Measured
- HMC on the Smets-Wouters posterior (36 parameters, 2 chains x 300 draws, finite-difference gradients):
  no divergences, max R-hat 1.014, all posterior means within one sd of Dynare's mode. It is **not**
  more efficient than random-walk Metropolis there: median effective sample size per 1000 posterior
  evaluations was 1.1 for HMC (each gradient costs about 73 evaluations) against 3.4 for the adaptive
  random walk (4 x 40000 draws, max R-hat 1.076). Exact gradients are needed for HMC to pay off.

### Fixed
- Edge-case audit: degenerate inputs now raise `ValidationError` instead of a raw numpy/Python error or a
  silent NaN: non-finite Kalman system matrices, empty chains/series/start points (`stationary_distribution`,
  `quantile`, `fixed_point`, `metropolis`, `hmc`), negative or non-finite Lorenz/Gini weights, non-finite
  `exp_grid` curvature, non-finite bootstrap data, and bad `klein_solve` matrices.

### Changed
- Benchmark record re-measured on 0.2.0 (`benchmarks/results/20261105_JayabrataBasu.json`); the
  EGM-vs-VFI 20x goal is documented as missed and retired. New `egm_cs_500` case (same grid as VFI).

## [0.2.0] - 2026-11-04

Headline: differentiable (exact-gradient) estimation of heterogeneous-agent models. Verified against
finite differences and on model-generated data; not yet tested on real data (planned for 0.3.0).

### Added
- `dynecon.estimation.sensitivity.stationary_moments`: exact Jacobian of stationary-distribution
  moments with respect to parameters (implicit differentiation of the EGM and Young-lottery
  equations; needs the `jax` extra). `equilibrium_moments` adds the market-clearing price via the
  implicit function theorem.
- `dynecon.estimation.smm_exact.smm_exact`: gradient-based SMM with those Jacobians (partial or
  general equilibrium). Tutorial 10 compares it with derivative-free `smm`.
- `smm_exact(starts=[...])` (multi-start) and a matrix-free GMRES path for models above 4000 states.
- `fixed_point(anderson=m)`: opt-in Anderson acceleration with safeguards.
- `AR1Shock(normalize=True)`: values divided by their stationary mean (unit-mean income).

### Changed
- EGM uses Anderson acceleration by default (`anderson=0` restores plain iteration): fewer
  iterations (about 2x at tolerance 1e-8, 20-35% less wall-clock time), same solution to the solver tolerance; it falls back to the plain iteration if the
  accelerated run fails. Provenance facts that cannot change within a process (including the git
  commit) are collected once instead of on every solve.
- Built-in `huggett` now has `rho`, `sigma_e` parameters and an `a_max` option like `aiyagari`.
- Built-in `aiyagari`: `rho` and `sigma_e` are now model parameters (so they can be estimated), the
  wage is computed from the parameter view instead of being fixed when the model is built, income is
  an `AR1Shock(normalize=True)`, and `a_max` can be fixed (`aiyagari.equilibrium(a_max=80.0)`).
  `aiyagari.excess_demand` is a JAX-traceable market-clearing function for `equilibrium_moments`.
  Numerical results are unchanged. `capital_demand` and `wage` no longer force `float`.

## [0.1.1] - 2026-11-04

### Changed
- Documentation is hosted at <https://dynecon.readthedocs.io>; project metadata and README link to it.
- CI fixes from the first public run: sweep workers use `spawn` (no fork after OpenMP),
  the `jax` availability check ignores a bare directory named `jax`, NumPy-stub typing fixes.
- GitHub Actions pinned by commit SHA.
- CI now tests Python 3.10 to 3.15 on Linux, macOS and Windows; added a
  minimum-dependency-versions job; classifiers and README state the supported range.

### Fixed
- `solve()` imported JAX (and Numba) just to record versions in provenance when they were installed;
  versions now come from package metadata, so the optional stacks load only when used.

## [0.1.0] - 2026-11-03

### Added
- **Core:** error hierarchy (`ValidationError`, `ModelSpecificationError`, `BackendError`,
  `ConvergenceError`/`ConvergenceWarning`, `DeterminacyError`, `IdentificationError`), `Solution`
  result type with `save`/`load`/`to_dict`/`policy_function`, RNG contract (`as_generator`,
  `spawn`), lazy `jit` decorator with pure-Python fallback, validation helpers, provenance.
- **Numerics:** uniform and curved grids, 1-D linear interpolation (NumPy, Numba, JAX) with
  clamp/linear extrapolation, Tauchen and Rouwenhorst, Gauss-Hermite/Legendre, Markov tools
  (exact GTH stationary distribution, chain simulation), the single `fixed_point` driver,
  numerical Jacobians.
- **Model DSL:** `saving`, `finite_dp`, `dsge` and `pricing` problem templates, AR(1)/Markov/IID
  shocks, `ParameterSet` with transforms, immutable `CompiledModel` with `with_params` and a
  SHA-256 fingerprint, collected validation errors.
- **Solvers:** `solve()` with vectorised and Numba VFI (Howard, monotone search), EGM, Euler
  errors, stationary-equilibrium engine, first-order perturbation (Klein/QZ) with IRFs and
  theoretical moments, Lucas-tree pricing, batched JAX EGM (`solve_many`).
- **Simulation:** panel simulation, Young histograms, moments and bootstrap helpers, Lorenz/Gini,
  generalised IRFs, linear simulation of DSGE solutions.
- **Estimation:** SMM (two-step, J-test), MLE with JAX gradients, Kalman filter and DSGE
  likelihood, adaptive Metropolis-Hastings with R-hat/ESS, priors, identification diagnostics.
- **Models:** consumption-savings, Brock-Mirman, RBC, McCall, Aiyagari, Huggett, two-period OLG,
  three-equation New Keynesian, Lucas tree.
- **Data, plots, experiments:** HP/Baxter-King/Hamilton filters, FRED parsing, 13 plot functions,
  `ParameterSweep`, `Experiment` with `reproduce()`, `PolicyExperiment` with CEV welfare,
  Monte Carlo and bootstrap, elasticities.
- **Engineering:** CI workflows, coverage and dead-code gates, benchmark harness with recorded
  results.

### Changed
- **Renamed to Dynecon** (`pip install dynecon`, `import dynecon`; was EconKit / `econkit`, a name taken on PyPI).
- **Licence is now BSD-3-Clause** (was GPL-3.0).
- Package layout rebuilt around the execution plan; version is single-sourced in
  `dynecon/_version.py`; SciPy is now a required dependency.
- `config.set_backend` validates names; default backend is `"auto"` (Numba when installed).

### Removed (git tag `v0.1.0-baseline` keeps the old code)
- `solvers/canonical.py` (pure-Python double-loop solvers and the liquid/illiquid prototype),
  `VFISolution`, `value_function_iteration`, `VFISolver`, convergence-artifact exporters,
  `simulate_markov_asset_policy`, `run_solve_simulate_moments_pipeline`, `core/types.py`,
  `core/protocols.py`, `Model.equation`/`EquationSpec`, JAX wrappers that only delegated to NumPy,
  `gauss_laguerre_exp`, `golden_section_max`, `brent_max`, `expectation_discrete`,
  `scripts/benchmark_solver_backends.py`.
