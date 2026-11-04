# Estimation

All estimators are functions returning immutable result objects (`smm(...) -> SMMResult`,
`mle(...) -> MLEResult`, `metropolis/bayes_estimate -> MCMCResult`). Options are validated and
recorded on the result.

## Simulated method of moments

```python
from dynecon.estimation import MomentSet, MomentTarget, smm

target = MomentSet((MomentTarget("mean_assets", 3.1), MomentTarget("std_consumption", 0.4)))
model = models.load("consumption_savings")  # then mark the parameter free with bounds
res = smm(
    model.compile(),
    ["beta"],
    None,
    target,
    n_obs=90_000,
    n_sim=1.0,
    solve_options={"tol": 1e-8},
    sim_options={"n_periods": 300, "n_agents": 300, "burn_in": 100},
    weights="optimal",
    rng=1,
)
print(res.summary())
```

For each candidate `theta` the model is recompiled, solved and simulated with **common random
numbers** (the same seed at every evaluation); the loss is `g'Wg`, `g = m(theta) - m_hat`.
The optimiser is `scipy.optimize.minimize` on an unconstrained transform of the parameters
(logit for two-sided bounds, log for a lower bound). Failed inner solves get a penalty and are
counted in `result.diagnostics["failed_evals"]`.

### Inference

With `G = dm/dtheta` (central differences, common random numbers; `diagnostics["jacobian_step_gap"]`
compares two step sizes), `Omega` the long-run covariance of the data moments (scaled by `N`;
estimated by simulation unless `omega=` is given), `S = n_sim`:

```
V = (1 + 1/S) (G'WG)^-1 G'W Omega W G (G'WG)^-1 / N          (reduces to (1+1/S)(G'Omega^-1 G)^-1 / N when W = Omega^-1)
J = N / (1 + 1/S) * g' Omega^-1 g  ~  chi2(q - p)             (over-identified systems only)
```

`weights="optimal"` is two-step: identity weights first, then `W = Omega^-1` at the first-step
estimate. **Calibration is exactly identified SMM**: with as many moments as parameters the
minimum loss is zero and the model hits the targets.

`identification_report(res)` reports the rank of the (elasticity-scaled) Jacobian and the
condition number of `G'WG`, lists weakly identified parameters, and raises `IdentificationError`
when it is rank-deficient. `loss_profile` traces the loss along one parameter.

## Exact-gradient SMM (stationary-distribution moments)

When the moments are computed from the model's **stationary distribution** (not from a simulated
sample) they are smooth, noise-free functions of the parameters, and Dynecon differentiates them
exactly: the household fixed point and the Young-lottery distribution are differentiated by the
implicit function theorem, and so is the market-clearing price. The forward solve is the usual
NumPy/Numba EGM solver; JAX is used only for the derivatives, so this needs the `jax` extra.

```python
import numpy as np
from dynecon import models
from dynecon.estimation.sensitivity import equilibrium_moments, stationary_moments
from dynecon.estimation.smm_exact import smm_exact
from dynecon.models import aiyagari

def moments(mu, c, a_next, a_grid, z, p):        # jax.numpy-compatible; mu, c, a_next: (assets, shocks)
    mean_a, mean_c = (mu * a_next).sum(), (mu * c).sum()
    return [mean_a, (mu * a_next**2).sum() - mean_a**2, (mu * c**2).sum() - mean_c**2]

grid = dict(n_a=40, n_z=3, a_max=80.0)           # fix the grid so the objective is smooth
names = ["beta", "sigma_e"]
eq = aiyagari.equilibrium(**grid, beta=0.96, sigma_e=0.2)

# Moments and their exact Jacobian at an equilibrium (also dr*/dtheta):
m, G, dr = equilibrium_moments(eq.model, names, "r", aiyagari.excess_demand, moments)

# Estimate: omega is the asymptotic covariance of sqrt(N) * (data moments), from the data.
omega = np.diag((0.05 * np.abs(m)) ** 2)
market = ("r", lambda mod: (-mod.p.delta + 1e-4, 1 / mod.p.beta - 1 - 1e-4), aiyagari.excess_demand)
start = models.load("aiyagari", **grid, beta=0.95, sigma_e=0.15).compile()
res = smm_exact(start, names, m, moments, omega, n_obs=10_000, equilibrium=market)
print(res.summary())
```

* `stationary_moments(model, names, moment_fn)` is the partial-equilibrium building block (a fixed
  interest rate); `equilibrium_moments(...)` adds `dprice/dtheta = -F_theta / F_price` for a model
  already solved at the clearing price and returns the *total* Jacobian.
* `smm_exact` uses L-BFGS-B with the exact gradient `2 G'Wg`, the bounds of the parameter set, and
  the same `SMMResult` as `smm` (J-test, standard errors with `n_sim = inf`, identification tools).
  Parameter values where the model cannot be solved get a large penalty (counted in
  `diagnostics["failed_evals"]`); a failure at the starting point raises.
* Equations must be JAX-traceable: the built-in utilities and the built-in `aiyagari` are; model
  functions should compute prices such as the wage from the parameter view `p` (as `aiyagari` does)
  and not capture them at build time. `AR1Shock` parameters (`rho`, `sigma_e`) are differentiable.
  The asset grid and any `MarkovShock` are held fixed.
* Size: up to 4000 states (for example 200 assets x 20 shocks) the implicit systems are solved
  densely; above that a matrix-free GMRES takes over (checked against finite differences at 6000
  states; the first call per grid shape also pays a one-off JAX compile of several seconds).
  `smm_exact(..., starts=[...])` also fits from extra start vectors and keeps the best.
* Limits: derivatives are piecewise smooth where the borrowing constraint starts to bind. Tutorial 10
  compares this estimator with the derivative-free `smm` (fewer model evaluations, same estimates).

## Maximum likelihood and Kalman filter

`mle(loglik, theta0, bounds)` maximises with `L-BFGS-B`; standard errors come from the inverse
of minus the numerical Hessian (`gradient="jax"` uses exact derivatives for traceable
likelihoods). `kalman_filter` handles missing observations (NaN entries are skipped);
`loglik_state_space(solution, observables, data, measurement_error)` builds the likelihood of a
perturbation solution.

## Bayesian estimation

`bayes_estimate(model, free_params, loglik, n_draws)` combines `Param.prior` (Normal, Uniform,
Beta, Gamma, InverseGamma) with a likelihood and samples with adaptive random-walk Metropolis
(Haario et al.) on the unconstrained scale, adding the log-Jacobian so the target is the
posterior of the original parameters. Check `result.rhat < 1.05` and `result.ess`.
