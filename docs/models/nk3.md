# nk3

Three-equation New Keynesian model.

**Economics.** A Phillips curve (inflation), a dynamic IS curve (output gap) and an interest-rate rule.
The Taylor principle (``phi_pi > 1`` when ``phi_x = 0``) is necessary for a unique
equilibrium; otherwise sunspot fluctuations are possible.

**Equations.** $\pi=\beta E\pi'+\kappa x$; $x=Ex'-(i-E\pi')/\sigma$; $i=\phi_\pi\pi+\phi_xx+v$.

## Calibration

Source: Textbook (Galí 2015 style) quarterly calibration.

| parameter | default | bounds | description |
|---|---:|---|---|
| `beta` | 0.99 | (0.5, 0.9999) | Discount factor |
| `kappa` | 0.15 | (0.001, 5.0) | Phillips curve slope |
| `sigma` | 1 | (0.1, 10.0) | Inverse EIS |
| `phi_pi` | 1.5 | (0.0, 10.0) | Taylor rule on inflation |
| `phi_x` | 0.125 | (0.0, 10.0) | Taylor rule on output gap |
| `rho_v` | 0.5 | (0.0, 0.999) | Policy shock persistence |

Structural options (not parameters): none.

## Solution method

Perturbation (Klein).

## Usage

```python
import dynecon as ek
from dynecon import models

sol = ek.solve(models.load('nk3').compile())
```

## Benchmark and test

**Benchmark:** With $\phi_x=0$: $x=bv$, $b=-1/(\sigma(1-\rho)+\kappa(\phi_\pi-\rho)/(1-\beta\rho))$, $\pi=\kappa b v/(1-\beta\rho)$; determinate iff $\kappa(\phi_\pi-1)+(1-\beta)\phi_x>0$.

**What the test checks:** Closed-form coefficients; determinacy grid; Bullard-Mitra case. (see `tests/models/` and `tests/solvers/perturbation/`).
