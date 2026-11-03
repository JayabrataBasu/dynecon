# huggett

Huggett (1993) bond economy: incomplete markets with a zero-net-supply bond.

**Economics.** Households trade a risk-free bond that is in zero net supply; the equilibrium interest
rate makes aggregate bond holdings zero. Precautionary saving demands lie below the
complete-markets rate, and a tighter borrowing limit lowers the equilibrium rate further.

**Equations.** Bonds in zero net supply; borrowing limit $-\phi$.

## Calibration

Source: Huggett (1993) style; textbook calibration (annual).

| parameter | default | bounds | description |
|---|---:|---|---|
| `beta` | 0.96 | (0.5, 0.9999) | Discount factor |
| `sigma` | 2 | (0.2, 10.0) | CRRA coefficient |

Structural options (not parameters): `r=0.02`, `rho=0.9`, `sigma_e=0.2`, `phi=2.0`, `n_a=200`, `n_z=7`.

## Solution method

EGM + Young distribution + Brent on the bond market.

## Usage

```python
import dynecon as ek
from dynecon import models
from dynecon.models import huggett
eq = huggett.equilibrium(phi=2.0)
```

## Benchmark and test

**Benchmark:** $r^*<1/\beta-1$; a tighter limit lowers $r^*$.

**What the test checks:** Bond market clears; wealth Gini exceeds income Gini. (see `tests/models/` and `tests/solvers/perturbation/`).
