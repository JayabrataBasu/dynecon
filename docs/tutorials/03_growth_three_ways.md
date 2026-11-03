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

# Stochastic growth three ways: VFI, EGM and perturbation

With log utility and full depreciation the Brock-Mirman model has the exact policy
`k' = alpha beta z k^alpha`, `c = (1 - alpha beta) z k^alpha`. We recover it with each method.

```{code-cell} ipython3
import numpy as np
import econkit as ek
from econkit import models
from econkit.models.brock_mirman import ANALYTIC

alpha, beta = 0.36, 0.96
vfi = ek.solve(models.load("brock_mirman", n_k=300).compile(), method="vfi", tol=1e-9)
egm = ek.solve(models.load("brock_mirman", n_k=200).compile(), method="egm")
pert = ek.solve(models.load("brock_mirman", problem="dsge").compile())
print(vfi.method, egm.method, pert.method)
```

```{code-cell} ipython3
k, z = vfi.model.grids["k"], vfi.model.shocks["z"].values
truth = ANALYTIC["capital_policy"](k[:, None], z[None, :], alpha, beta)
h = np.diff(k).max()
ok = (truth > k[0] + 2 * h) & (truth < k[-1] - 2 * h)
vfi_err = np.abs(vfi.policy["k_next"] - truth)[ok].max()

ke, ze = egm.model.grids["k"], egm.model.shocks["z"].values
c_true = ANALYTIC["consumption_policy"](ke[:, None], ze[None, :], alpha, beta)
inside = (egm.policy["k_next"] > ke[0]) & (egm.policy["k_next"] < ke[-1])
egm_err = np.abs(egm.policy["c"] / c_true - 1)[inside].max()
print("VFI max error", vfi_err, "(grid spacing", h, ")   EGM max relative error", egm_err)
print("perturbation:  k' = %.4f k + %.4f z" % tuple(pert.P[0]), "  c = %.4f k + %.4f z" % tuple(pert.F[0]))
```

```{code-cell} ipython3
assert vfi_err <= 2 * h
assert egm_err < 1e-8
np.testing.assert_allclose(pert.P[0], [alpha, 1.0], atol=1e-6)
np.testing.assert_allclose(pert.F[0], [alpha, 1.0], atol=1e-6)
```
