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

# Quickstart: a consumption-savings model in 15 lines

A household with CRRA preferences earns stochastic income, saves in one asset and cannot
borrow. We load the built-in model, solve it, plot the policy and simulate a panel.

```{code-cell} ipython3
import numpy as np
import econkit as ek
from econkit import models

model = models.load("consumption_savings", n_a=100, n_z=5).compile()
sol = ek.solve(model)                     # EGM: the model supplies marginal utility
print(sol.summary())
```

The consumption policy against assets, one line per income state:

```{code-cell} ipython3
import matplotlib
matplotlib.use("Agg")
from econkit.viz import plot_policy
ax = plot_policy(sol, "consumption")
```

Simulate 200 households for 300 periods and look at moments:

```{code-cell} ipython3
sim = ek.simulate(sol, n_periods=300, n_agents=200, burn_in=100, rng=0)
print(sim.summary())
moments = sim.moments(["mean_assets", "std_consumption", "autocorr_income"])
moments
```

```{code-cell} ipython3
assert sol.converged and sol.method == "egm"
assert 0 < moments["mean_assets"] < 50
assert 0.8 < moments["autocorr_income"] < 1.0
```
