# brock_mirman

Brock-Mirman stochastic growth: log utility, Cobb-Douglas, full depreciation.

**Economics.** With log utility and full depreciation the planner's problem has a closed form: a constant
fraction ``alpha * beta`` of output is invested, ``k' = alpha beta z k^alpha``. It is the
canonical test case for every global and local solution method.

**Equations.** $k'=\alpha\beta zk^\alpha$, $c=(1-\alpha\beta)zk^\alpha$ with log utility and full depreciation.

## Calibration

Source: Brock and Mirman (1972); textbook calibration.

| parameter | default | bounds | description |
|---|---:|---|---|
| `alpha` | 0.36 | (0.05, 0.95) | Capital share |
| `beta` | 0.96 | (0.5, 0.9999) | Discount factor |
| `rho` | 0.9 | (0.0, 0.99) | TFP persistence (log AR(1)) |
| `sigma_e` | 0.02 | (0.001, 0.5) | TFP innovation std |

Structural options (not parameters): `problem='saving'`, `n_k=200`, `n_z=7`.

## Solution method

VFI, EGM (`saving` form) and perturbation (`problem='dsge'`).

## Usage

```python
import dynecon as ek
from dynecon import models

sol = ek.solve(models.load('brock_mirman', problem='dsge').compile())
```

## Benchmark and test

**Benchmark:** The closed-form policy above (exact for every $z$ process).

**What the test checks:** VFI within two grid spacings, EGM to 1e-10, perturbation coefficients $(\alpha,1)$ to 1e-10. (see `tests/models/` and `tests/solvers/perturbation/`).
