---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# McCall job search

An unemployed worker draws a wage offer each period, accepts (employed forever at that wage)
or rejects and collects the benefit `c`. The optimal rule is a reservation wage.

```{code-cell} ipython3
import numpy as np
import dynecon as ek
from dynecon import models
from dynecon.models import mccall

def wbar(**kw):
    return mccall.reservation_wage(ek.solve(models.load("mccall", **kw).compile(), tol=1e-12))

base = wbar()
exact = mccall.reservation_wage_uniform(0.96, 0.5)
print("reservation wage: numerical %.4f   closed form %.4f   grid spacing %.4f" % (base, exact, 1 / 59))
```

Comparative statics: higher benefits and more patience raise the reservation wage.

```{code-cell} ipython3
stats = {"c=0.4": wbar(c=0.4), "c=0.6": wbar(c=0.6), "beta=0.95": wbar(beta=0.95), "beta=0.97": wbar(beta=0.97)}
stats
```

```{code-cell} ipython3
assert abs(base - exact) <= 1 / 59 + 1e-12
assert stats["c=0.6"] > base > stats["c=0.4"]
assert stats["beta=0.97"] > base > stats["beta=0.95"]
```
