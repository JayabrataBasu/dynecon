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

# Estimating a heterogeneous-agent model with exact gradients

Moments of the model's *stationary distribution* (mean and variance of assets, variance of
consumption) are smooth deterministic functions of the parameters. Dynecon differentiates the
household fixed point and the stationary distribution exactly (implicit function theorem), so
the estimator gets the moment Jacobian for free instead of by repeated finite differences.
Needs the `jax` extra.

```{code-cell} ipython3
import time
import numpy as np
import dynecon as ek
from dynecon import models
from dynecon.estimation import smm
from dynecon.estimation.sensitivity import stationary_moments
from dynecon.estimation.smm_exact import smm_exact
from dynecon.simulation import stationary_distribution_young

def moment_fn(mu, c, a_next, a_grid, z, p):          # written with jax.numpy-compatible operations
    mean_a, mean_c = (mu * a_next).sum(), (mu * c).sum()
    return [mean_a, (mu * a_next**2).sum() - mean_a**2, (mu * c**2).sum() - mean_c**2]

names = ["beta", "sigma_e"]
truth = {"beta": 0.95, "sigma_e": 0.12}
start = models.load("consumption_savings", n_a=40, n_z=3, beta=0.93, sigma_e=0.08).compile()

# "Data" moments from the true parameters; Omega would come from the micro data (here: 5% sd).
m_hat, G = stationary_moments(start.with_params(**truth), names, moment_fn)
omega = np.diag((0.05 * np.abs(m_hat)) ** 2)
print("data moments:", m_hat.round(4))
print("exact Jacobian:\n", G.round(3))
```

## Exact-gradient estimation

```{code-cell} ipython3
t0 = time.perf_counter()
exact = smm_exact(start, names, m_hat, moment_fn, omega, n_obs=5000, weights="identity")
t_exact = time.perf_counter() - t0
print(exact.summary())
print(f"{exact.diagnostics['n_evals']} model evaluations, {t_exact:.1f} s")
```

## The same problem with derivative-free search

The existing `smm` with Nelder-Mead and finite-difference standard errors, on the very same
moments computed by the NumPy pipeline.

```{code-cell} ipython3
def numpy_moments(model, seed):
    sol = ek.solve(model, method="egm", tol=1e-13, max_iter=100_000)
    mu = stationary_distribution_young(sol, tol=1e-13)
    a_next, c = sol.policy["assets_next"], sol.policy["consumption"]
    mean_a, mean_c = (mu * a_next).sum(), (mu * c).sum()
    return np.array([mean_a, (mu * a_next**2).sum() - mean_a**2, (mu * c**2).sum() - mean_c**2])

t0 = time.perf_counter()
nm = smm(start, names, m_hat, numpy_moments, n_obs=5000, n_sim=1e9, omega=omega, weights="identity")
t_nm = time.perf_counter() - t0
print(f"{nm.diagnostics['n_evals']} model evaluations, {t_nm:.1f} s")
print("exact  :", exact.estimates.round(6), exact.std_errors.round(6))
print("Nelder-Mead:", nm.estimates.round(6), nm.std_errors.round(6))
```

Fewer model evaluations is the point. Wall-clock time here is similar because the first JAX call
compiles for several seconds and each exact evaluation costs a few forward solves; the advantage
grows with the number of parameters (finite differences need one extra solve per parameter).

```{code-cell} ipython3
true = np.array([truth[n] for n in names])
assert np.allclose(exact.estimates, true, rtol=1e-4)
assert np.allclose(nm.estimates, true, rtol=5e-3)
assert exact.diagnostics["n_evals"] < nm.diagnostics["n_evals"]
assert np.allclose(exact.std_errors, nm.std_errors, rtol=5e-2)
```
