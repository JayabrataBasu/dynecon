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

# RBC and New Keynesian models by perturbation; filtering business cycles

```{code-cell} ipython3
import numpy as np
import dynecon as ek
from dynecon import models
from dynecon.solvers.perturbation import irf, theoretical_moments
from dynecon.core.errors import DeterminacyError

rbc = ek.solve(models.load("rbc").compile())
resp = irf(rbc, "eps_z", horizon=40)
peak = int(np.argmax(resp["y"]))
print("output peaks", peak, "periods after the TFP shock; response", round(float(resp["y"][peak]), 4))
mom = theoretical_moments(rbc)
print("std of output / std of consumption:", mom.std[rbc.variables.index("y")] / mom.std[rbc.variables.index("c")])
```

Determinacy in the New Keynesian model (Taylor principle):

```{code-cell} ipython3
outcome = {}
for phi_pi in (0.5, 0.9, 1.1, 1.5, 3.0):
    try:
        ek.solve(models.load("nk3", phi_x=0.0, phi_pi=phi_pi).compile())
        outcome[phi_pi] = "determinate"
    except DeterminacyError as exc:
        outcome[phi_pi] = str(exc).split(":")[0]
outcome
```

Compare three detrending filters on simulated RBC output (log deviations):

```{code-cell} ipython3
from dynecon.data import hp_filter, baxter_king, hamilton_filter
sim = ek.simulate(rbc, n_periods=240, rng=1, store=["y"])
y = sim.paths["y"][:, 0]
trend, cycle = hp_filter(y, 1600)
bk = baxter_king(y)
_, ham = hamilton_filter(y)
print("std: HP %.4f  BK %.4f  Hamilton %.4f" % (np.std(cycle), np.std(bk), np.std(ham)))
```

A small FRED-format sample (synthetic values, no licensed data) parsed and HP-filtered:

```{code-cell} ipython3
from pathlib import Path
from dynecon.data import parse_fred_csv
text = Path("data/fred_sample.csv").read_text() if Path("data/fred_sample.csv").exists() else Path("docs/tutorials/data/fred_sample.csv").read_text()
gdp = parse_fred_csv(text).dropna()
t, c = hp_filter(np.log(gdp))
print(len(gdp), "observations; cycle std", float(c.std()))
```

```{code-cell} ipython3
assert peak <= 3
assert outcome[0.5] != "determinate" and outcome[0.9] != "determinate"
assert outcome[1.1] == outcome[1.5] == outcome[3.0] == "determinate"
assert len(bk) == 240 - 24 and len(ham) == 240 - 8 - 4 + 1
assert len(gdp) > 20
```
