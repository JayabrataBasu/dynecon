# Changelog

All notable changes are recorded here ([Keep a Changelog](https://keepachangelog.com)).

## [Unreleased]

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
