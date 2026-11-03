# rbc

Real business cycle model with inelastic labour.

**Economics.** A representative household consumes and accumulates capital; productivity follows a log
AR(1). The model has a known steady state (``k/y = alpha / (1/beta - 1 + delta)``) and is
the workhorse of quantitative macroeconomics, solved here by perturbation or EGM.

**Equations.** Euler $c^{-\sigma}=\beta E[c'^{-\sigma}(\alpha z'k'^{\alpha-1}+1-\delta)]$, $k'=(1-\delta)k+i$, $y=zk^\alpha$.

## Calibration

Source: Textbook quarterly calibration (e.g. Cooley and Prescott 1995 style).

| parameter | default | bounds | description |
|---|---:|---|---|
| `alpha` | 0.33 | (0.05, 0.95) | Capital share |
| `beta` | 0.99 | (0.5, 0.9999) | Discount factor |
| `delta` | 0.025 | (0.001, 1.0) | Depreciation rate |
| `sigma` | 1 | (0.2, 10.0) | CRRA coefficient (1 = log) |
| `rho` | 0.95 | (0.0, 0.999) | TFP persistence |
| `sigma_e` | 0.007 | (0.0001, 0.5) | TFP innovation std |

Structural options (not parameters): `problem='dsge'`, `n_k=200`, `n_z=7`.

## Solution method

Perturbation (default); EGM via `problem='saving'`.

## Usage

```python
import dynecon as ek
from dynecon import models

sol = ek.solve(models.load('rbc').compile()); irf = irf(sol, 'eps_z')
```

## Benchmark and test

**Benchmark:** Steady state $k/y=\alpha/(1/\beta-1+\delta)$, $i/y=\delta k/y$.

**What the test checks:** Steady-state identities, $\delta=1$ reproduces Brock-Mirman, perturbation vs EGM near the steady state, IRF shape. (see `tests/models/` and `tests/solvers/perturbation/`).
