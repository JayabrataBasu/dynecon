# Cross-check against QuantEcon

Generated from `crosscheck/results/quantecon.json` by `scripts/gen_crosscheck_doc.py`;
regenerate the data with `python scripts/crosscheck_quantecon.py`. Only numbers in this
file may be quoted elsewhere.

- Versions: dynecon 0.1.1, quantecon 0.13.1, numpy 2.5.4, python 3.14.8

[QuantEcon](https://quantecon.org) is an established open-source library for computational economics. Agreement with an independent implementation on identical inputs is evidence that Dynecon's results are right; a disagreement is either a documented convention difference or a bug to fix.

## Markov discretization (Tauchen, Rouwenhorst): agrees within tolerance

Compared: dynecon.numerics.tauchen/rouwenhorst vs quantecon.markov.tauchen/rouwenhorst.

80 parameter combinations.

| quantity | largest abs. difference | tolerance | worst case |
|---|---|---|---|
| grid | 0.00e+00 | 1e-12 | - |
| P | 6.66e-16 | 1e-12 | tauchen, rho=0.99, sigma_e=0.05, n=15 |
| stationary | 2.14e-10 | 1e-08 | tauchen, rho=0.99, sigma_e=0.05, n=5 |

- Argument order differs: Dynecon (rho, sigma_e, n_states), QuantEcon (n, rho, sigma).
- Both take sigma as the innovation standard deviation and use the same grid (+-3 unconditional sd for Tauchen, +-sd*sqrt(n-1) for Rouwenhorst).
- Stationary distributions differ most for near-unit-root chains (rho=0.99), where the problem is ill-conditioned; this is numerical conditioning, not a convention gap.

## Kalman filter: agrees within tolerance

Compared: dynecon.estimation.kalman_filter vs quantecon.Kalman (+ scipy likelihood).

20 parameter combinations.

| quantity | largest abs. difference | tolerance | worst case |
|---|---|---|---|
| predicted_state | 3.89e-15 | 1e-10 | - |
| predicted_cov | 3.33e-15 | 1e-10 | - |
| filtered_state | 4.44e-15 | 1e-10 | - |
| loglik | 1.14e-13 | 1e-10 | - |

- Random stable systems (1-4 states, 1-3 observables, 80 periods, fixed seed).
- Dynecon takes covariances (H, Q); QuantEcon takes factors (C, H) with covariances CC' and HH'. Both treat the initial (a0, P0) as the prior for the first period.
- The log-likelihood includes the full Gaussian constant.
