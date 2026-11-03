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

# Policy experiment and welfare: relaxing a borrowing limit

We compare a household economy where borrowing is forbidden with one where households can
borrow up to 2 units, and measure the welfare change as a consumption-equivalent variation.
This is a **partial-equilibrium** comparison of stationary economies (prices are fixed).

```{code-cell} ipython3
import numpy as np
from econkit import models
from econkit.model import State
from econkit.experiment import PolicyExperiment

base = models.load("consumption_savings", n_a=120, n_z=5, a_max=40.0)

def allow_borrowing(m):
    m = m.copy("borrowing")
    s = m.get_state("assets")
    m.replace_state(State("assets", bounds=(-2.0, s.bounds[1]), grid_type="exp",
                          n_points=s.n_points, grid_curvature=s.grid_curvature))
    return m

out = PolicyExperiment(base, allow_borrowing).run()
print("CEV: %.3f%% of baseline consumption" % (100 * out.welfare["cev"]))
for regime in ("baseline", "reform"):
    print(regime, {k: round(v, 3) for k, v in out.aggregates[regime].items()})
```

Welfare is evaluated on the baseline stationary cross-section (the households alive when the
reform arrives); transition costs are out of scope.

```{code-cell} ipython3
higher_r = PolicyExperiment(base, {"r": 0.04}).run()
print("CEV of a higher interest rate: %.3f%%" % (100 * higher_r.welfare["cev"]))
```

```{code-cell} ipython3
assert out.welfare["cev"] > 0
assert higher_r.welfare["cev"] > 0
assert out.aggregates["reform"]["mean_assets"] < out.aggregates["baseline"]["mean_assets"]
```
