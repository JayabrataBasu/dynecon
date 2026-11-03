# olg2

Two-period overlapping-generations model (Diamond): log utility, Cobb-Douglas, full depreciation.

**Economics.** Young agents save a constant share of their wage; the capital stock per young worker
follows an exact one-dimensional law of motion with a unique, globally stable steady state.

**Equations.** $(1+n)k'=\frac{\beta}{1+\beta}(1-\alpha)k^\alpha$.

## Calibration

Source: Diamond (1965); one period = about 30 years (beta = 0.96^30, n = 1.01^30 - 1).

| parameter | default | bounds | description |
|---|---:|---|---|
| `alpha` | 0.3 | (0.05, 0.95) | Capital share |
| `beta` | 0.293858 | (0.01, 1.0) | Per-generation discount factor |
| `n` | 0.347849 | (-0.5, 5.0) | Population growth per period |

Structural options (not parameters): none.

## Solution method

Exact law of motion; perturbation for the slope; Euler root-finding as a check.

## Usage

```python
import econkit as ek
from econkit import models

sol = ek.solve(models.load("olg2").compile())
```

## Benchmark and test

**Benchmark:** $k^*=[\beta(1-\alpha)/((1+\beta)(1+n))]^{1/(1-\alpha)}$, globally stable.

**What the test checks:** Steady state three ways to 1e-10; convergence from three starting points. (see `tests/models/` and `tests/solvers/perturbation/`).
