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
