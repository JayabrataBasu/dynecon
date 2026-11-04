# aiyagari

Aiyagari (1994) economy: incomplete markets, idiosyncratic income risk, capital market.

**Economics.** Households self-insure against uninsurable income risk by holding capital. Precautionary
saving pushes the equilibrium interest rate below the complete-markets rate ``1/beta - 1``,
and more income risk lowers it further.

**Equations.** Household as consumption-savings with wage $w(r)$; firm $r=\alpha(K/L)^{\alpha-1}-\delta$.

## Calibration

Source: Aiyagari (1994) Table II style (sigma=3, rho=0.6, sigma_e=0.2); textbook.

| parameter | default | bounds | description |
|---|---:|---|---|
| `beta` | 0.96 | (0.5, 0.9999) | Discount factor |
| `sigma` | 3 | (0.2, 10.0) | CRRA coefficient |
| `alpha` | 0.36 | (0.05, 0.95) | Capital share |
| `delta` | 0.08 | (0.0, 1.0) | Depreciation |
| `rho` | 0.6 | (0.0, 0.99) | Income persistence |
| `sigma_e` | 0.2 | (0.01, 1.0) | Income innovation sd |

Structural options (not parameters): `r=0.03`, `n_a=200`, `n_z=7`, `a_max=None`.

## Solution method

EGM + Young distribution + Brent on the capital market.

## Usage

```python
import dynecon as ek
from dynecon import models
from dynecon.models import aiyagari
eq = aiyagari.equilibrium(sigma=2.0, rho=0.9, sigma_e=0.2)
```

## Benchmark and test

**Benchmark:** $r^*<1/\beta-1$ strictly; $K/Y=\alpha/(r^*+\delta)$.

**What the test checks:** Market clearing, $r^*$ decreasing in income risk, warm starts save iterations. (see `tests/models/` and `tests/solvers/perturbation/`).
