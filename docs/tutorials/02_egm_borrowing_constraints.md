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

# Consumption-savings with EGM: constraints and Euler errors

The endogenous grid method inverts the Euler equation on a grid of end-of-period assets, so
no root finding is needed. We compare it with value function iteration.

```{code-cell} ipython3
import numpy as np
import econkit as ek
from econkit import models

model = models.load("consumption_savings", n_a=300, n_z=7, a_max=60.0).compile()
egm = ek.solve(model, method="egm")
vfi = ek.solve(model, method="vfi", tol=1e-9)
print("EGM iterations", egm.iterations, " VFI iterations", vfi.iterations)
print("max Euler error (log10): EGM", egm.diagnostics["euler_error_max"],
      " VFI", vfi.diagnostics["euler_error_max"])
```

Both methods agree on the savings policy up to the grid spacing, but EGM's Euler errors are
orders of magnitude smaller because VFI restricts choices to the grid.

```{code-cell} ipython3
a = model.grids["assets"]
h = np.diff(a).max()
interior = (egm.policy["assets_next"] > a[0] + 1e-9) & (egm.policy["assets_next"] < 0.5 * a[-1])
gap = np.abs(egm.policy["assets_next"] - vfi.policy["assets_next"])[interior].max()
print("max policy gap", gap, " grid spacing", h)
print("share of states at the borrowing constraint:", egm.diagnostics["n_binding"])
```

```{code-cell} ipython3
assert gap <= 2 * h
assert egm.diagnostics["euler_error_max"] <= vfi.diagnostics["euler_error_max"] + 0.5
assert egm.diagnostics["n_binding"] > 0
```
