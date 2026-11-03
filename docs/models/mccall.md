# mccall

McCall job-search model (finite-state dynamic programming).

**Economics.** An unemployed worker draws a wage offer each period and either accepts (employed at that
wage forever) or rejects and collects the benefit ``c``. The optimal rule is a reservation
wage: accept iff the offer exceeds it. Higher benefits or patience raise the reservation
wage.

**Equations.** $\bar w=c+\frac{\beta}{1-\beta}E[(w'-\bar w)^+]$.

## Calibration

Source: McCall (1970); textbook calibration with uniform offers on [0, 1].

| parameter | default | bounds | description |
|---|---:|---|---|
| `beta` | 0.96 | (0.5, 0.9999) | Discount factor |
| `c` | 0.5 | (0.0, 2.0) | Unemployment benefit |

Structural options (not parameters): `n_w=60`.

## Solution method

Tabular VFI on a `finite_dp` problem.

## Usage

```python
import dynecon as ek
from dynecon import models
from dynecon.models import mccall
w = mccall.reservation_wage(ek.solve(models.load('mccall').compile()))
```

## Benchmark and test

**Benchmark:** Uniform offers: $k\bar w^2-(2k+1)\bar w+(k+c)=0$, $k=\beta/(2(1-\beta))$.

**What the test checks:** Reservation wage within one grid spacing; comparative statics in $c$ and $\beta$; threshold rule. (see `tests/models/` and `tests/solvers/perturbation/`).
