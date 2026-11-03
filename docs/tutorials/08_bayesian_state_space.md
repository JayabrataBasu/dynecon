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

# Bayesian estimation of a state-space model (Kalman filter + Metropolis)

A local-level model: `y_t = mu_t + e_t`, `mu_{t+1} = mu_t + eta_t`. We place priors on the two
log-variances, evaluate the exact Gaussian likelihood with a (JAX, jitted) Kalman filter and
sample the posterior with adaptive Metropolis-Hastings.

```{code-cell} ipython3
import numpy as np
import jax, jax.numpy as jnp
from econkit.estimation import metropolis
from econkit.estimation.kalman_jax import kalman_loglik_jax

rng = np.random.default_rng(0)
true_h, true_q, T = 0.5, 0.1, 300
mu = np.cumsum(np.sqrt(true_q) * rng.standard_normal(T))
y = mu + np.sqrt(true_h) * rng.standard_normal(T)
Z, Tm = jnp.ones((1, 1)), jnp.ones((1, 1))

@jax.jit
def loglik(theta):
    H = jnp.exp(theta[0]) * jnp.ones((1, 1))
    Q = jnp.exp(theta[1]) * jnp.ones((1, 1))
    return kalman_loglik_jax(y, Z, H, Tm, Q, jnp.zeros(1), 10.0 * jnp.ones((1, 1)))

def logpost(theta):                      # N(0, 2^2) priors on the log variances
    return float(loglik(jnp.asarray(theta))) - 0.5 * float(np.sum(np.asarray(theta) ** 2) / 4.0)

res = metropolis(logpost, [0.0, -1.0], 1500, n_chains=3, burn_in=500, rng=1, names=["log_h", "log_q"])
print(res.summary())
print("posterior mean of h, q:", np.exp(res.mean))
```

```{code-cell} ipython3
import matplotlib
matplotlib.use("Agg")
from econkit.viz import plot_mcmc_trace, plot_posterior
ax = plot_mcmc_trace(res, "log_h")
ax2 = plot_posterior(res, "log_q")
```

```{code-cell} ipython3
assert np.all(res.rhat < 1.1)
assert abs(res.mean[0] - np.log(true_h)) < 0.6 and abs(res.mean[1] - np.log(true_q)) < 1.0
```
