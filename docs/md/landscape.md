# Landscape: where Dynecon stands (research of 2026-11-04)

Internal planning note (excluded from the Sphinx build). Feeds issue #10 (choose the headline 1.0
feature). **Verified** = read from the package's PyPI metadata or its GitHub README on the date above.
Anything marked *not stated* means the README did not mention it; that is **not** evidence the feature
is absent. Re-check before quoting publicly.

## Packages compared

| Package | Latest (PyPI) | Python | Licence | Compute stack | What it is (per its README/PyPI) |
|---|---|---|---|---|---|
| **Dynecon** | 0.2.0 (2026-11-04) | >=3.10 | BSD-3 | NumPy/SciPy, optional Numba, optional JAX | Model DSL; VFI/EGM household solvers; stationary equilibrium; Klein perturbation; simulation; SMM, MLE, Kalman, MCMC |
| **QuantEcon.py** | 0.13.1 (2026-10-17) | >=3.12 | BSD (not in metadata) | Numba, NumPy, SciPy | Toolbox of algorithms and lecture code (DiscreteDP, Kalman, LQ, Markov tools). No household/Aiyagari class (checked in 0.13.1) |
| **Econ-ARK / HARK** | 0.17.2 (2026-05-01) | >=3.10 | Apache-2.0 | Numba, NumPy, pandas | Toolkit for heterogeneous-agent structural models, simulation-based aggregates. README calls itself beta and "not extensively tested" |
| **Sequence-Space Jacobian** | 1.0.0 (2022-01-10, only PyPI release) | >=3.7 | MIT | Numba, NumPy, SciPy (no JAX stated) | HetBlocks/DAG models, linearized and nonlinear IRFs, SSJ method; a Krusell-Smith notebook computes a log-likelihood |
| **econpizza** | 0.6.10 (2026-06-22) | *not stated* | MIT | JAX | Nonlinear perfect-foresight models with or without heterogeneous agents (HANK, one/two assets); autodiff Newton. Estimation *not stated* |
| **dolo** | 0.4.9.20 (2024-11-07) | <3.12 pinned | BSD-2 | NumPy, pandas<2 pinned | YAML models; perturbation, time iteration, VFI. Heterogeneous agents, autodiff, estimation *not stated* |
| **gEconpy** | 2.4.0 (2026-10-16) | >=3.12 | GPL | PyMC, Numba | gEcon-style DSGE with PyMC Bayesian estimation (representative agent); docs once described it as alpha |
| **Dynare 6.x** | manual 6.4 (2025-06) | MATLAB/Octave | GPL | n/a | Mature DSGE: perturbation, Bayesian and moment-based estimation. Heterogeneous agents and a Python interface *not verified* |

## Capability map (verified or stated only)

| Capability | Dynecon | QuantEcon | HARK | SSJ | econpizza | dolo | gEconpy | Dynare |
|---|---|---|---|---|---|---|---|---|
| Household savings solvers (VFI/EGM) | yes | DiscreteDP only | yes | via HetBlock | yes (HA) | VFI | no | no |
| Stationary HA equilibrium | yes | no | simulation-based | yes | yes | not stated | no | no |
| Aggregate shocks with heterogeneity (HANK) | **no** | no | not stated | **yes** | **yes** (perfect foresight) | not stated | no | not verified |
| DSGE perturbation | yes (1st order) | LQ | no | linearized via SSJ | no | up to 3rd order | yes | yes |
| Exact derivatives (JAX) | optional, partial | no | not stated | no | **yes** | no | PyMC/numba | n/a |
| Likelihood / Bayesian estimation | Kalman MLE, MCMC | Kalman recursions | not stated | log-likelihood example | not stated | not stated | **Bayesian (PyMC)** | **yes** |
| Simulation-based estimation (SMM) | yes | no | not stated | not stated | not stated | not stated | no | moment-based |
| Cross-checked vs independent library | **yes** (see crosscheck.md) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Observations

1. The established players cover pieces: SSJ owns heterogeneous agents with aggregate shocks;
   econpizza owns JAX-based nonlinear HA models; gEconpy and Dynare own Bayesian DSGE; QuantEcon is the
   teaching/toolbox standard; HARK is the broad structural-agent toolkit.
2. Nothing found combines, in one pip-installable package: heterogeneous-agent household solvers,
   stationary equilibrium, *and* gradient-based estimation (SMM/likelihood/HMC) with exact derivatives
   through the equilibrium. SSJ estimates but without autodiff; econpizza has autodiff but no stated
   estimation. This is a gap in what was *read*, not proof of absence (READMEs only).
3. Head-on competition with SSJ on HANK aggregate shocks would be expensive (it is the reference
   implementation of the method) and SSJ's last PyPI release is 2022, so interoperability may be a
   better route than reimplementation.
4. Maintenance is a differentiator in itself: dolo is pinned to Python <3.12 and pandas <2; SSJ's last
   release is 2022; HARK self-describes as lightly tested. Dynecon is tested on Python 3.10-3.15 on
   three OSes, at its dependency floors, and against QuantEcon.

## Candidate headline features (for issue #10)

| Candidate | Fit with existing code | Differentiation | Cost/risk |
|---|---|---|---|
| **A. Differentiable estimation of heterogeneous-agent models**: gradients of stationary-equilibrium moments w.r.t. parameters (implicit function theorem on the market-clearing condition), enabling gradient-based SMM and HMC | High: JAX backend, SMM, MCMC, Kalman-gradient MLE and stationary equilibrium already exist | High: no README-level evidence of this combination elsewhere | Medium: needs JAX-traceable household solve + implicit differentiation + parity tests vs finite differences |
| B. Sequence-space Jacobians / HANK aggregate shocks | Medium: builds on distribution/Jacobian pieces | Low-medium: SSJ is the reference | High |
| C. Richer built-in model library and calibration workflows | High | Low: easy to copy | Low |

**Recommendation: A**, scoped as "gradient-based estimation of a heterogeneous-agent model", validated
against finite differences and against the existing SMM results. It plays to what is already built,
fits the macro-time-series application planned for Phase 4, and avoids competing head-on with SSJ.
The decision is the maintainer's.

## Measured results (2026-11-04, see crosscheck.md for the full tables)

Installed and benchmarked on identical problems (one Windows machine): SSJ and QuantEcon only.
- Household EGM + stationary distribution vs SSJ: results agree (policy ~2e-7, distribution ~3e-9) but
  **SSJ is still faster**, by about 1.5x to 2x (Dynecon/SSJ time ratio 1.5x to 2.0x with Numba at matched
  tolerances, interleaved runs in one process: SSJ 0.020 s vs Dynecon 0.04 s at 200 assets). The 0.2.0 solver changes
  (Anderson-accelerated EGM) cut Dynecon's own time by about 20-35% (iterations roughly halve) but do not close the gap.
  Likely cause (not profiled): per-iteration Python overhead in Dynecon's loop. One machine; see crosscheck.md for the table.
- Discrete savings problem vs QuantEcon `DiscreteDP`: identical answers; Dynecon VFI is 12x to 29x
  faster than DiscreteDP policy iteration (more against value iteration) on 180 to 450 states (0.2.0 run; 0.1.1
  measured 2.0x to 8.7x, the difference is likely the per-solve `git` call removed in 0.2.0, not profiled separately).
- Not benchmarked: HARK (no identical problem set up yet), econpizza, dolo, gEconpy, Dynare.
- Implication: speed is not a differentiator against SSJ; correctness evidence, packaging and the
  differentiable-estimation feature are.

## Caveats on this research

The capability table is README/PyPI-level only. Only SSJ and QuantEcon were installed and benchmarked.
Dynare's heterogeneous-agent and Python support was not verified. Re-verify every "not stated" before
publishing any comparison.

## Sources

- https://pypi.org/project/quantecon/, https://pypi.org/project/econ-ark/, https://pypi.org/project/sequence-jacobian/,
  https://pypi.org/project/econpizza/, https://pypi.org/project/dolo/, https://pypi.org/project/gEconpy/ (PyPI metadata via JSON API)
- https://github.com/shade-econ/sequence-jacobian, https://github.com/gboehl/econpizza,
  https://github.com/econ-ark/HARK, https://github.com/EconForge/dolo (READMEs)
- https://www.dynare.org/new-dynare-release/dynare-6.0-released/, https://www.dynare.org/manual.pdf
- https://geconpy.readthedocs.io (gEconpy)
