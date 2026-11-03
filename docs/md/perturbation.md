# First-order perturbation (Klein 2000)

For a `dsge` problem `E_t f(y_{t+1}, y_t; p) = 0` EconKit finds the steady state
(analytic `steady_state(p)` or `scipy.optimize.root` from `guess(p)`), linearises by
central differences (or exact JAX autodiff with `backend="jax"`), and solves
`A E_t[yhat_{t+1}] = B yhat_t` with the generalised Schur decomposition.

## Writing equations

```python
m.set_problem(
    "dsge",
    variables=["k", "z", "c"],
    states=["k", "z"],
    shocks=["eps"],
    log_variables=["k", "z", "c"],  # report log deviations
    equations=lambda yp, y, p: [
        yp.k - y.z * y.k**p.alpha + y.c,  # resource constraint
        log(yp.z) - p.rho * log(y.z),  # TFP; log from econkit.models.utility
        1 / y.c - p.beta * p.alpha * yp.z * yp.k ** (p.alpha - 1) / yp.c,  # Euler
    ],
    shock_loadings=lambda p: np.array([[0.0], [p.sigma_e]]),  # (n_states x n_shocks)
    steady_state=lambda p: ...,
)
```

Predetermined variables come first in `variables`. `yp` is next period (`t+1`), `y` this period;
the expectation operator is implicit. `shock_loadings` maps structural shocks into innovations of
the predetermined block.

## Solution

`sol = ek.solve(model)` returns a `PerturbationSolution` with `policy["F"]` (jump variables
`u = F x`), `policy["P"]` (`x' = P x + Sigma e'`), `policy["Sigma"]`. Helpers:
`irf(sol, shock, size, horizon)`, `theoretical_moments(sol, lags)` (Lyapunov covariance,
autocovariances, variance decomposition), `simulate_linear` (also reached through
`ek.simulate`).

## Determinacy message decoder

| Message | Meaning |
|---|---|
| `indeterminate: n_stable=a > n_predetermined=b` | too many stable roots: sunspot equilibria (e.g. a Taylor rule that is too passive) |
| `no stable solution (explosive): n_stable < n_predetermined` | too few stable roots |
| `Z11 is (numerically) singular` | degenerate stable subspace; check the equations |

Unit roots count as unstable (a random-walk state raises the explosive error). The three-equation
NK model is determinate iff `kappa (phi_pi - 1) + (1 - beta) phi_x > 0`.

Second-order perturbation is not in v1.
