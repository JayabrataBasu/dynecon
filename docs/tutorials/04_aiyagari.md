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

# Aiyagari: precautionary saving and the equilibrium interest rate

Households cannot insure income risk and self-insure by holding capital. In general
equilibrium the interest rate falls below the complete-markets rate `1/beta - 1`.

```{code-cell} ipython3
import numpy as np
from dynecon.models import aiyagari
from dynecon.simulation import gini, marginal_distribution

kw = dict(sigma=2.0, rho=0.9, n_a=100, n_z=5)
eq = aiyagari.equilibrium(sigma_e=0.2, **kw)
print("equilibrium r* = %.4f%%   (complete markets: %.4f%%)" % (100 * eq.price, 100 * (1 / 0.96 - 1)))
print({k: round(v, 4) for k, v in eq.aggregates.items()})
```

More risk means more precautionary saving and a lower rate:

```{code-cell} ipython3
rates = {s: aiyagari.equilibrium(sigma_e=s, **kw).price for s in (0.05, 0.1, 0.2)}
rates
```

The wealth distribution is far more unequal than income:

```{code-cell} ipython3
import matplotlib
matplotlib.use("Agg")
from dynecon.viz import plot_lorenz
a = eq.grids["assets"]
wealth_mass = marginal_distribution(eq.distribution, 0)
ax = plot_lorenz(a, wealth_mass)
wealth_gini = gini(a, wealth_mass)
income_gini = gini(eq.model.shocks["income"].values, marginal_distribution(eq.distribution, 1))
print("Gini: wealth %.3f, income %.3f" % (wealth_gini, income_gini))
```

```{code-cell} ipython3
assert eq.price < 1 / 0.96 - 1
assert rates[0.05] > rates[0.1] > rates[0.2]
assert abs(eq.aggregates["capital_supply"] - eq.aggregates["capital_demand"]) < 1e-6 * eq.aggregates["capital_demand"]
assert wealth_gini > income_gini
```
