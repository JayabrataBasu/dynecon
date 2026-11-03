# EconKit v1.0 - Progress

Checklist only. No percentages (R6). Tick a box only when the phase gates pass.


## P0 — Baseline repair, hygiene and lean tooling

- [ ] P0.1 Install this plan and the progress tracker
- [ ] P0.2 Correct the record
- [ ] P0.3 Packaging
- [ ] P0.4 pytest and coverage configuration
- [ ] P0.5 Track and create CI (D3)
- [ ] P0.6 PyPI name check (D5)
- [ ] P0.7 Backend roadmap document
- [ ] P0.8 Housekeeping files
- [ ] P0.9 Quality baseline
- [ ] P0.10 Pin the repo state
- [ ] P0.11 Phase extractor (context saving)
- [ ] P0.12 Lean tooling and baseline audit (R21)

## P1 — Core contract: errors, RNG, JIT, config

- [ ] P1.1 Error hierarchy (`core/errors.py`)
- [ ] P1.3 RNG contract (`utils/rng.py`)
- [ ] P1.4 JIT decorator with fallback (`utils/jit.py`)
- [ ] P1.5 Validation helpers (`utils/validation.py`)
- [ ] P1.7 Config and backend introspection
- [ ] P1.8 Core cleanup and API policy
- [ ] P1.9 Provenance helper (`utils/provenance.py`)

## P2 — Numerics I: grids and 1-D interpolation

- [ ] P2.1 Grid constructors (cold path, NumPy only)
- [ ] P2.2 1-D linear interpolation: harden and accelerate
- [ ] P2.7 Registration and dispatch coverage
- [ ] P2.9 Math doc and docstrings
- [ ] P2.10 Property-based tests (Hypothesis)

## P3 — Numerics II: the iteration driver, Jacobians, Markov tools, quadrature

- [ ] P3.2 Numerical Jacobian
- [ ] P3.3 The fixed-point iteration driver
- [ ] P3.4 Markov-chain utilities
- [ ] P3.6 Quadrature addition
- [ ] P3.8 Remaining baseline numerics audit
- [ ] P3.9 Property-based tests (Hypothesis)

## P4 — Model specification system

- [ ] P4.1 Shocks and state spaces
- [ ] P4.2 Parameter sets
- [ ] P4.3 Problem templates (replaces `Model.equation`)
- [ ] P4.5 Validation
- [ ] P4.6 Introspection and fingerprint
- [ ] P4.8 Port existing model tests
- [ ] P4.9 Docs
- [ ] P4.10 Builder hygiene

## P5 — Solver core: `solve()`, `Solution`, vectorized and Numba VFI

- [ ] P5.1 `Solution`, the `Solver` protocol and `solve()`
- [ ] P5.2 Payoff tensor and NumPy Bellman reference (`saving` problems)
- [ ] P5.3 Numba Bellman kernel (hot path)
- [ ] P5.4 Howard acceleration and monotone search
- [ ] P5.5 Delete the baseline solvers and port their tests (resolves TD1, TD2, TD8)
- [ ] P5.6 Diagnostics
- [ ] P5.7 Verbosity
- [ ] P5.8 Analytic validation: Brock–Mirman via VFI
- [ ] P5.9 Analytic validation: permanent income
- [ ] P5.10 Performance acceptance (measured, not asserted)

## P6 — EGM

- [ ] P6.1 EGM algorithm (infinite horizon, any `saving` problem with `marginal_utility`, `inverse_marginal_utility`, `marginal_return`)
- [ ] P6.2 Monotone endogenous grid
- [ ] P6.4 Euler-equation errors
- [ ] P6.5 Analytic validation: Brock–Mirman and permanent income via EGM
- [ ] P6.6 Cross-method consistency
- [ ] P6.7 Wire-up
- [ ] P6.8 Golden-file regression tests

## P7 — Simulation engine and distributions

- [ ] P7.1 `SimulationResult` and `simulate()`
- [ ] P7.2 Moments helpers (`simulation/moments.py`)
- [ ] P7.3 Young (2010) histogram method
- [ ] P7.4 Aggregates and inequality
- [ ] P7.5 Impulse responses from nonlinear models (policy-based)
- [ ] P7.7 Determinism contract tests
- [ ] P7.8 Docs
- [ ] P7.9 Port check
- [ ] P7.10 Property-based tests (Hypothesis)

## P8 — Equilibrium solvers

- [ ] P8.1 Generic stationary-equilibrium engine
- [ ] P8.2 Aiyagari implementation requirements (used by P10.5)
- [ ] P8.3 Huggett implementation requirements
- [ ] P8.4 Analytic/structural tests
- [ ] P8.5 Performance sanity
- [ ] P8.6 Docs

## P9 — Perturbation (first-order, Klein/QZ)

- [ ] P9.1 Problem specification (adds the `dsge` template to `set_problem`, P4.3)
- [ ] P9.2 Steady state
- [ ] P9.3 Linearisation
- [ ] P9.4 Klein (2000) solution (defines `DeterminacyError` and `PerturbationConfig`)
- [ ] P9.5 Impulse responses and moments
- [ ] P9.6 Analytic validations
- [ ] P9.7 Interop
- [ ] P9.8 Docs
- [ ] P9.9 Property-based and golden-file tests

## P10 — Built-in model library

- [ ] P10.0 Uniform interface (do this first)
- [ ] P10.1 `consumption_savings` (income-fluctuation problem)
- [ ] P10.2 `brock_mirman` (stochastic growth, log utility, full depreciation)
- [ ] P10.3 `rbc` (real business cycle, inelastic labour)
- [ ] P10.4 `mccall` (job search)
- [ ] P10.5 `aiyagari` (P8 engine)
- [ ] P10.6 `huggett` (bond economy, P8 engine)
- [ ] P10.7 `olg2` (two-period overlapping generations, log utility, Cobb–Douglas, full depreciation, population growth `n`)
- [ ] P10.8 `nk3` (three-equation New Keynesian)
- [ ] P10.9 `lucas_tree` (asset pricing, endowment economy)
- [ ] P10.10 Cross-model integration test
- [ ] P10.11 Model registry docs
- [ ] P10.12 Extension point
- [ ] P10.13 External cross-validation (optional)

## P11 — Estimation

- [ ] P11.1 Parameter transforms
- [ ] P11.2 Moment targets (extend existing)
- [ ] P11.3 SMM
- [ ] P11.4 SMM recovery tests (with closed-form model)
- [ ] P11.5 MLE
- [ ] P11.6 Kalman filter and linear-Gaussian likelihood
- [ ] P11.7 Bayesian estimation (random-walk Metropolis–Hastings)
- [ ] P11.9 Identification diagnostics
- [ ] P11.10 Delete the demo pipeline

## P12 — JAX backend and automatic differentiation

- [ ] P12.1 Setup and policy
- [ ] P12.2 Operations to implement (each with NumPy parity test at 1e-10 under x64)
- [ ] P12.3 Autodiff in perturbation
- [ ] P12.4 Gradients for estimation
- [ ] P12.5 Vectorisation across parameters (consumer: `ParameterSweep`, P14.1)
- [ ] P12.7 Dispatch semantics
- [ ] P12.8 CI

## P13 — Data and visualization

- [ ] P13.3 Filters (cold path, NumPy/SciPy; array-like in, arrays out, index preserved when given a `pandas.Series`)
- [ ] P13.5 FRED download and parse
- [ ] P13.7 Visualization infrastructure
- [ ] P13.8 The plot set (each with a smoke test using the `Agg` backend, and each used by at least one tutorial or model page — `check_usage.py` enforces it)
- [ ] P13.9 Import hygiene test

## P14 — Experiments and reproducibility

- [ ] P14.1 `ParameterSweep`
- [ ] P14.2 `Solution` persistence, `Experiment` and the provenance store
- [ ] P14.3 Counterfactual policy experiments
- [ ] P14.4 Welfare: consumption-equivalent variation (`welfare.py`)
- [ ] P14.5 Monte Carlo and bootstrap
- [ ] P14.6 Elasticities
- [ ] P14.8 Docs

## P15 — Performance engineering and benchmarks

- [ ] P15.1 Harness
- [ ] P15.2 Benchmark cases (name → description)
- [ ] P15.3 Record and publish
- [ ] P15.4 Targets and honest reporting
- [ ] P15.5 Profiling-driven fixes (time-boxed)
- [ ] P15.6 Regression detection
- [ ] P15.7 Import-time and footprint budget

## P16 — Documentation and tutorials

- [ ] P16.1 Sphinx scaffold
- [ ] P16.2 User guide
- [ ] P16.3 Tutorials (nine, each executes in ≤ 60 s; each ends with an assertion cell that checks a key number so broken tutorials fail the docs build)
- [ ] P16.4 Model pages and theory pages
- [ ] P16.5 API reference and docstring audit
- [ ] P16.6 Honest README
- [ ] P16.7 Contributor docs
- [ ] P16.8 Build and publish docs
- [ ] P16.9 Documentation review pass
- [ ] P16.10 JOSS paper preparation
- [ ] P16.11 Known limits page

## P17 — Release engineering and v1.0.0

- [ ] P17.1 API freeze
- [ ] P17.2 Licence and metadata (needs D1, D8)
- [ ] P17.3 Final quality sweep
- [ ] P17.4 Build and clean-room install
- [ ] P17.5 Release candidate (human-gated)
- [ ] P17.6 Soak
- [ ] P17.7 Final changelog and release notes
- [ ] P17.8 Final release (human-gated)
- [ ] P17.9 Post-release hygiene
- [ ] P17.10 Retrospective
- [ ] P17.11 Supply-chain hygiene

## Parking lot

(none)

## Blockers

(none)
