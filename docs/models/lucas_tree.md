# lucas_tree

Lucas (1978) tree: asset pricing in an endowment economy.

**Economics.** A representative household consumes the dividend of a tree. With log utility the
price-dividend ratio is constant at ``beta / (1 - beta)`` whatever the dividend process:
income and substitution effects of dividend news exactly offset.

**Equations.** $p(y)u'(y)=\beta E[u'(y')(p(y')+y')]$.

## Calibration

Source: Lucas (1978); textbook log-AR(1) dividend calibration.

| parameter | default | bounds | description |
|---|---:|---|---|
| `beta` | 0.96 | (0.5, 0.9999) | Discount factor |
| `sigma` | 1 | (0.2, 10.0) | CRRA coefficient (1 = log) |
| `rho` | 0.9 | (0.0, 0.99) | Dividend persistence (log AR(1)) |
| `sigma_e` | 0.05 | (0.001, 0.5) | Dividend innovation std |

Structural options (not parameters): `n_z=9`.

## Solution method

Fixed point of the price functional.

## Usage

```python
import dynecon as ek
from dynecon import models

sol = ek.solve(models.load('lucas_tree', sigma=3.0).compile())
```

## Benchmark and test

**Benchmark:** Log utility: $p(y)/y=\beta/(1-\beta)$ for any dividend process.

**What the test checks:** Constant price-dividend ratio; sign of its slope for $\sigma\gtrless1$; risk-free rate two ways. (see `tests/models/` and `tests/solvers/perturbation/`).
