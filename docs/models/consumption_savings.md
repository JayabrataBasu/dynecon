# consumption_savings

Income-fluctuation (consumption-savings) problem with a borrowing limit.

**Economics.** A household with CRRA preferences earns stochastic income, saves in one risk-free asset
at rate ``r`` and cannot borrow below ``a_min``. Precautionary saving and the
borrowing constraint make consumption concave in wealth.

**Equations.** $V(a,z)=\max_{a'\ge a_{min}}u(c)+\beta E[V(a',z')]$, $c=(1+r)a+z-a'$; $\ln z$ AR(1).

## Calibration

Source: Textbook calibration (annual); not tied to a specific paper.

| parameter | default | bounds | description |
|---|---:|---|---|
| `beta` | 0.96 | (0.5, 0.9999) | Discount factor |
| `sigma` | 2 | (0.2, 10.0) | CRRA coefficient |
| `r` | 0.03 | (-0.1, 0.2) | Net interest rate |
| `rho` | 0.9 | (0.0, 0.99) | Income persistence (log AR(1)) |
| `sigma_e` | 0.1 | (0.001, 1.0) | Income innovation std |

Structural options (not parameters): `a_min=0.0`, `a_max=50.0`, `n_a=200`, `n_z=7`, `curvature=3.0`.

## Solution method

EGM (default); VFI for comparison.

## Usage

```python
import dynecon as ek
from dynecon import models

sol = ek.solve(models.load('consumption_savings').compile())
```

## Benchmark and test

**Benchmark:** Permanent-income limit ($\beta(1+r)=1$, no risk: $a'=a$); the infinite-horizon MPC $1-\beta^{1/\sigma}(1+r)^{(1-\sigma)/\sigma}$ (informational).

**What the test checks:** Policy identity under permanent income; concavity of $c(a)$; the constraint binds only at low wealth and income. (see `tests/models/` and `tests/solvers/perturbation/`).
