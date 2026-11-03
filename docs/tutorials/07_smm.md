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

# Estimating a model with simulated method of moments

We recover the persistence and volatility of an AR(1) from its variance and autocorrelation.
(The "model" here is pure simulation; the same call works for any solved model by passing a
`MomentSet` instead of a function, see the estimation guide.)

```{code-cell} ipython3
import numpy as np
from scipy.signal import lfilter
from dynecon.estimation import smm, identification_report
from dynecon.model import Model, Param

m = Model("ar1")
m.params(rho=Param(0.5, bounds=(0.0, 0.999), free=True), sigma=Param(1.0, bounds=(0.01, 5.0), free=True),
         beta=Param(0.9))
m.set_problem("finite_dp", reward=lambda p: np.ones((1, 1)), transition=lambda p: np.ones((1, 1, 1)),
              discount="beta")
model = m.compile()

N, S = 5000, 20
def path(rho, sigma, n, seed):
    return lfilter([sigma], [1.0, -rho], np.random.default_rng(seed).standard_normal(n))
def moments(y):
    return np.array([y.var(), np.corrcoef(y[1:], y[:-1])[0, 1], np.corrcoef(y[2:], y[:-2])[0, 1]])

data = moments(path(0.9, 0.5, N, 123))
fn2 = lambda mod, seed: moments(path(mod.p.rho, mod.p.sigma, S * N, seed))[:2]
```

**Exactly identified (calibration):** two moments, two parameters; the model hits the targets.

```{code-cell} ipython3
cal = smm(model, ["rho", "sigma"], data[:2], fn2, n_obs=N, n_sim=S, rng=0)
print(cal.summary())
print("rank of the moment Jacobian:", identification_report(cal).rank)
```

**Over-identified:** add the second autocorrelation; the J-test checks the extra restriction.

```{code-cell} ipython3
fn3 = lambda mod, seed: moments(path(mod.p.rho, mod.p.sigma, S * N, seed))
est = smm(model, ["rho", "sigma"], data, fn3, n_obs=N, n_sim=S, weights="optimal", rng=0)
print(est.summary())
```

```{code-cell} ipython3
assert cal.loss < 1e-10
assert abs(cal.params["rho"] - 0.9) < 4 * cal.std_errors[0]
assert abs(est.params["sigma"] - 0.5) < 4 * est.std_errors[1]
assert est.df == 1 and est.j_pvalue > 0.01
```
