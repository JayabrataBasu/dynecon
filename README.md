# Dynecon

[![PyPI](https://img.shields.io/pypi/v/dynecon)](https://pypi.org/project/dynecon/)
[![Docs](https://readthedocs.org/projects/dynecon/badge/?version=latest)](https://dynecon.readthedocs.io)
[![CI](https://github.com/JayabrataBasu/dynecon/actions/workflows/ci.yml/badge.svg)](https://github.com/JayabrataBasu/dynecon/actions/workflows/ci.yml)

A Python library for computational economics: specify, solve, simulate and estimate dynamic
economic models, with a NumPy reference implementation, optional Numba acceleration and an
optional JAX backend for exact derivatives.

> **Status: pre-1.0 (0.1.1).** Published on PyPI; the API may still change before 1.0.
> Documentation: <https://dynecon.readthedocs.io>. Progress is tracked as a checklist
> in [`docs/md/V1_PROGRESS.md`](docs/md/V1_PROGRESS.md); known limits are in
> [`docs/md/known_limits.md`](docs/md/known_limits.md).

## Install

Requires Python 3.10 or newer (tested on 3.10 to 3.15). On macOS the system `python3` is 3.9, so use a
virtual environment with a newer Python: `python3.12 -m venv .venv && source .venv/bin/activate`.

```bash
pip install dynecon               # from PyPI
pip install -e ".[dev]"           # or from a clone, for development
```

Runtime dependencies are NumPy and SciPy. Extras: `numba`, `jax`, `data` (pandas), `viz`
(matplotlib), `docs`, `dev`, `all`. `import dynecon` loads none of the optional stacks.

## Example

```python
import dynecon as ek
from dynecon import models

model = models.load("consumption_savings").compile()  # CRRA household, borrowing limit, income risk
sol = ek.solve(model)  # endogenous grid method (EGM)
print(sol.summary())  # converged, residual, Euler errors, fingerprint

sim = ek.simulate(sol, n_periods=500, n_agents=1000, burn_in=100, rng=0)
print(sim.moments(["mean_assets", "std_consumption", "autocorr_income"]))

# first-order perturbation of a DSGE model and an impulse response
from dynecon.solvers.perturbation import irf

rbc = ek.solve(models.load("rbc").compile())
print(irf(rbc, "eps_z", horizon=8)["y"])
```

## What is in the library

| Area | Contents |
|---|---|
| Model DSL | `saving`, `finite_dp`, `dsge`, `pricing` templates; AR(1)/Markov/IID shocks; immutable compiled models with `with_params` and a fingerprint |
| Solvers | `solve()`: value function iteration (vectorised + Numba, Howard, monotone search), endogenous grid method, first-order perturbation (Klein/QZ), Lucas-tree pricing, stationary-equilibrium engine, batched JAX EGM (`solve_many`) |
| Simulation | panel simulation, Young histograms, Lorenz/Gini, generalised IRFs, linear simulation of DSGE solutions |
| Estimation | SMM (two-step, J-test, standard errors), MLE, Kalman filter and DSGE likelihood, adaptive Metropolis-Hastings (R-hat, ESS), priors, identification diagnostics |
| Models | consumption-savings, Brock-Mirman, RBC, McCall, Aiyagari, Huggett, two-period OLG, three-equation New Keynesian, Lucas tree |
| Data and plots | HP / Baxter-King / Hamilton filters, FRED CSV parsing, 13 matplotlib plot functions |
| Experiments | parameter sweeps, reproducible `Experiment` runs, policy experiments with consumption-equivalent welfare, Monte Carlo, bootstrap, elasticities |

Every model is tested against an analytic benchmark where one exists (Brock-Mirman policy,
permanent income, the McCall reservation-wage quadratic, the NK closed form and Taylor principle,
the Diamond OLG steady state, the log-utility Lucas price-dividend ratio).

## Backends

| Backend | Use | Status |
|---|---|---|
| `numpy` | reference, debugging | all operations |
| `numba` | CPU speed (default when installed) | interpolation, Bellman maximisation, simulation, histogram step, chain simulation |
| `jax` | exact derivatives, `vmap`, GPU | interpolation, Bellman, histogram, chain simulation, autodiff Jacobians, gradients, batched EGM |

## Measured performance

From [`docs/md/benchmarks.md`](docs/md/benchmarks.md) (minimum of repeats; Intel 12 logical cores,
Windows, Python 3.14, NumPy 2.5, Numba 0.68, JAX 0.12; one machine, never compare across machines):

| Case | NumPy | Numba | Speedup |
|---|---:|---:|---:|
| one Bellman maximisation, 1000 assets x 7 shocks | 119 ms | 3.9 ms | 30x |
| full VFI solve, 500 x 7 | 2.0 s | 0.26 s | 7.7x |
| panel simulation, 2x10^4 agents x 500 periods | 11.0 s | 0.55 s | 20x |
| 200 histogram steps, 1000 x 7 | 132 ms | 31 ms | 4.2x |

One Bellman iteration of the pre-rewrite pure-Python solver took 1.01 s at 200 x 7; the Numba
kernel takes 0.25 ms. Honest misses are documented too: EGM is *more accurate* than VFI but, at
the default 1e-10 tolerance, not faster than Numba VFI (see the benchmark page). JAX is not
faster than Numba on CPU.

## Documentation

Full documentation: <https://dynecon.readthedocs.io>.

- Guides: [`docs/md/`](docs/md) (DSL, backends, simulation, equilibrium, perturbation, estimation, experiments).
- Ten executable tutorials: [`docs/tutorials/`](docs/tutorials).
- Theory pages: [`docs/theory/`](docs/theory).
- Architecture and execution plan: [`docs/md/Dynecon_Architecture.md`](docs/md/Dynecon_Architecture.md),
  [`docs/md/V1_EXECUTION_PLAN.md`](docs/md/V1_EXECUTION_PLAN.md).
- API stability policy: [`docs/md/API_POLICY.md`](docs/md/API_POLICY.md).

Build the Sphinx site with `pip install -e ".[docs,jax]" && python scripts/gen_model_docs.py &&
sphinx-build -b html docs docs/_build/html`.

## What is not in v1

Time iteration, projection methods, Krusell-Smith and transition paths, HANK, second-order
perturbation, continuous-time models, HMC, a SymPy front end, plotly, and C/Rust layers are
post-1.0 (plan Part E). See [`docs/md/known_limits.md`](docs/md/known_limits.md).

## Development

```bash
make install && make ci     # layout + progress checks, ruff, mypy, pytest, dead-code gate
make coverage               # coverage report
make bench                  # python -m benchmarks.run --all
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Contributions follow rule R21: nothing without a caller
and a test.

## Citing

See [`CITATION.cff`](CITATION.cff).

## Licence

BSD 3-Clause (see `LICENSE`), (c) 2026 Jayabrata Basu.
