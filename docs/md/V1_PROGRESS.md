# EconKit v1.0 - Progress

Checklist only. No percentages (R6). Tick a box only when the phase gates pass.


## P0 — Baseline repair, hygiene and lean tooling

- [x] P0.1 Install this plan and the progress tracker
- [x] P0.2 Correct the record
- [x] P0.3 Packaging
- [x] P0.4 pytest and coverage configuration
- [x] P0.5 Track and create CI (D3)
- [x] P0.6 PyPI name check (D5)
- [x] P0.7 Backend roadmap document
- [x] P0.8 Housekeeping files
- [x] P0.9 Quality baseline
- [x] P0.10 Pin the repo state
- [x] P0.11 Phase extractor (context saving)
- [x] P0.12 Lean tooling and baseline audit (R21)

## P1 — Core contract: errors, RNG, JIT, config

- [x] P1.1 Error hierarchy (`core/errors.py`)
- [x] P1.3 RNG contract (`utils/rng.py`)
- [x] P1.4 JIT decorator with fallback (`utils/jit.py`)
- [x] P1.5 Validation helpers (`utils/validation.py`)
- [x] P1.7 Config and backend introspection
- [x] P1.8 Core cleanup and API policy
- [x] P1.9 Provenance helper (`utils/provenance.py`)

## P2 — Numerics I: grids and 1-D interpolation

- [x] P2.1 Grid constructors (cold path, NumPy only)
- [x] P2.2 1-D linear interpolation: harden and accelerate
- [x] P2.7 Registration and dispatch coverage
- [x] P2.9 Math doc and docstrings
- [x] P2.10 Property-based tests (Hypothesis)

## P3 — Numerics II: the iteration driver, Jacobians, Markov tools, quadrature

- [x] P3.2 Numerical Jacobian
- [x] P3.3 The fixed-point iteration driver
- [x] P3.4 Markov-chain utilities
- [x] P3.6 Quadrature addition
- [x] P3.8 Remaining baseline numerics audit
- [x] P3.9 Property-based tests (Hypothesis)

## P4 — Model specification system

- [x] P4.1 Shocks and state spaces
- [x] P4.2 Parameter sets
- [x] P4.3 Problem templates (replaces `Model.equation`)
- [x] P4.5 Validation
- [x] P4.6 Introspection and fingerprint
- [x] P4.8 Port existing model tests
- [x] P4.9 Docs
- [x] P4.10 Builder hygiene

## P5 — Solver core: `solve()`, `Solution`, vectorized and Numba VFI

- [x] P5.1 `Solution`, the `Solver` protocol and `solve()`
- [x] P5.2 Payoff tensor and NumPy Bellman reference (`saving` problems)
- [x] P5.3 Numba Bellman kernel (hot path)
- [x] P5.4 Howard acceleration and monotone search
- [x] P5.5 Delete the baseline solvers and port their tests (resolves TD1, TD2, TD8)
- [x] P5.6 Diagnostics
- [x] P5.7 Verbosity
- [x] P5.8 Analytic validation: Brock–Mirman via VFI
- [x] P5.9 Analytic validation: permanent income
- [x] P5.10 Performance acceptance (measured, not asserted)

## P6 — EGM

- [x] P6.1 EGM algorithm (infinite horizon, any `saving` problem with `marginal_utility`, `inverse_marginal_utility`, `marginal_return`)
- [x] P6.2 Monotone endogenous grid
- [x] P6.4 Euler-equation errors
- [x] P6.5 Analytic validation: Brock–Mirman and permanent income via EGM
- [x] P6.6 Cross-method consistency
- [x] P6.7 Wire-up
- [x] P6.8 Golden-file regression tests

## P7 — Simulation engine and distributions

- [x] P7.1 `SimulationResult` and `simulate()`
- [x] P7.2 Moments helpers (`simulation/moments.py`)
- [x] P7.3 Young (2010) histogram method
- [x] P7.4 Aggregates and inequality
- [x] P7.5 Impulse responses from nonlinear models (policy-based)
- [x] P7.7 Determinism contract tests
- [x] P7.8 Docs
- [x] P7.9 Port check
- [x] P7.10 Property-based tests (Hypothesis)

## P8 — Equilibrium solvers

- [x] P8.1 Generic stationary-equilibrium engine
- [x] P8.2 Aiyagari implementation requirements (used by P10.5)
- [x] P8.3 Huggett implementation requirements
- [x] P8.4 Analytic/structural tests
- [x] P8.5 Performance sanity
- [x] P8.6 Docs

## P9 — Perturbation (first-order, Klein/QZ)

- [x] P9.1 Problem specification (adds the `dsge` template to `set_problem`, P4.3)
- [x] P9.2 Steady state
- [x] P9.3 Linearisation
- [x] P9.4 Klein (2000) solution (defines `DeterminacyError` and `PerturbationConfig`)
- [x] P9.5 Impulse responses and moments
- [x] P9.6 Analytic validations
- [x] P9.7 Interop
- [x] P9.8 Docs
- [x] P9.9 Property-based and golden-file tests

## P10 — Built-in model library

- [x] P10.0 Uniform interface (do this first)
- [x] P10.1 `consumption_savings` (income-fluctuation problem)
- [x] P10.2 `brock_mirman` (stochastic growth, log utility, full depreciation)
- [x] P10.3 `rbc` (real business cycle, inelastic labour)
- [x] P10.4 `mccall` (job search)
- [x] P10.5 `aiyagari` (P8 engine)
- [x] P10.6 `huggett` (bond economy, P8 engine)
- [x] P10.7 `olg2` (two-period overlapping generations, log utility, Cobb–Douglas, full depreciation, population growth `n`)
- [x] P10.8 `nk3` (three-equation New Keynesian)
- [x] P10.9 `lucas_tree` (asset pricing, endowment economy)
- [x] P10.10 Cross-model integration test
- [x] P10.11 Model registry docs
- [x] P10.12 Extension point
- [ ] P10.13 External cross-validation (optional)

## P11 — Estimation

- [x] P11.1 Parameter transforms
- [x] P11.2 Moment targets (extend existing)
- [x] P11.3 SMM
- [x] P11.4 SMM recovery tests (with closed-form model)
- [x] P11.5 MLE
- [x] P11.6 Kalman filter and linear-Gaussian likelihood
- [x] P11.7 Bayesian estimation (random-walk Metropolis–Hastings)
- [x] P11.9 Identification diagnostics
- [x] P11.10 Delete the demo pipeline

## P12 — JAX backend and automatic differentiation

- [x] P12.1 Setup and policy
- [x] P12.2 Operations to implement (each with NumPy parity test at 1e-10 under x64)
- [x] P12.3 Autodiff in perturbation
- [x] P12.4 Gradients for estimation
- [x] P12.5 Vectorisation across parameters (consumer: `ParameterSweep`, P14.1)
- [x] P12.7 Dispatch semantics
- [x] P12.8 CI

## P13 — Data and visualization

- [x] P13.3 Filters (cold path, NumPy/SciPy; array-like in, arrays out, index preserved when given a `pandas.Series`)
- [x] P13.5 FRED download and parse
- [x] P13.7 Visualization infrastructure
- [x] P13.8 The plot set (each with a smoke test using the `Agg` backend, and each used by at least one tutorial or model page — `check_usage.py` enforces it)
- [x] P13.9 Import hygiene test

## P14 — Experiments and reproducibility

- [x] P14.1 `ParameterSweep`
- [x] P14.2 `Solution` persistence, `Experiment` and the provenance store
- [x] P14.3 Counterfactual policy experiments
- [x] P14.4 Welfare: consumption-equivalent variation (`welfare.py`)
- [x] P14.5 Monte Carlo and bootstrap
- [x] P14.6 Elasticities
- [x] P14.8 Docs

## P15 — Performance engineering and benchmarks

- [x] P15.1 Harness
- [x] P15.2 Benchmark cases (name → description)
- [x] P15.3 Record and publish
- [x] P15.4 Targets and honest reporting
- [ ] P15.5 Profiling-driven fixes (time-boxed)
- [x] P15.6 Regression detection
- [x] P15.7 Import-time and footprint budget

## P16 — Documentation and tutorials

- [x] P16.1 Sphinx scaffold
- [x] P16.2 User guide
- [x] P16.3 Tutorials (nine, each executes in ≤ 60 s; each ends with an assertion cell that checks a key number so broken tutorials fail the docs build)
- [x] P16.4 Model pages and theory pages
- [x] P16.5 API reference and docstring audit
- [x] P16.6 Honest README
- [x] P16.7 Contributor docs
- [ ] P16.8 Build and publish docs
- [ ] P16.9 Documentation review pass
- [x] P16.10 JOSS paper preparation
- [x] P16.11 Known limits page

## P17 — Release engineering and v1.0.0

- [x] P17.1 API freeze
- [x] P17.2 Licence and metadata (needs D1, D8)
- [x] P17.3 Final quality sweep
- [x] P17.4 Build and clean-room install
- [ ] P17.5 Release candidate (human-gated)
- [ ] P17.6 Soak
- [ ] P17.7 Final changelog and release notes
- [ ] P17.8 Final release (human-gated)
- [ ] P17.9 Post-release hygiene
- [ ] P17.10 Retrospective
- [ ] P17.11 Supply-chain hygiene

## Parking lot

- EGM is more accurate than Numba VFI but not faster at the default 1e-10 tolerance (361 contraction
  steps at ~2 ms); consider Anderson acceleration or a looser default tolerance (post-1.0 / Part E).
- P15.5: profile the three slowest benchmark cases (`aiyagari_ge`, `smm_ar1`, `simulate_panel` NumPy
  reference), not just EGM.
- P10.13: optional QuantEcon cross-validation of Tauchen/Rouwenhorst and stationary distributions.
- Pin GitHub Actions by commit SHA and configure PyPI trusted publishing (P17.11).
- Run the CI matrix on GitHub (Python 3.10-3.12; Linux/macOS/Windows); only Python 3.14 on Windows has
  been exercised locally.
- Verify the FRED `fredgraph.csv` endpoint against the live service.
- `simulate_panel` benchmark uses 2x10^4 agents, not 10^5 (memory).

## Blockers

- **PyPI name `econkit` is taken** (existing project, versions up to 0.5.8); the maintainer chose to keep it, so an upload under this name will be refused unless the name is obtained.
- D1 licence decided: BSD-3-Clause. D8 author metadata set (Jayabrata Basu).
- Human gates: Review-2/3/4, release candidate, soak, final release.
- Ticked boxes mean the step's code, tests and docs exist and the local gates passed on Python 3.14 /
  Windows; steps that mention CI (P0.5, P12.8) mean the files were created, **not** that they ran on
  GitHub.
