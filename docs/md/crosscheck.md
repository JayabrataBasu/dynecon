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

## Household savings problem (VFI): agrees within tolerance

Compared: dynecon.solve(method='vfi') + stationary_distribution_young vs quantecon.markov.DiscreteDP (policy iteration) + controlled_mc stationary distribution.

3 parameter combinations.

| quantity | largest abs. difference | tolerance | worst case |
|---|---|---|---|
| value | 1.55e-12 | 1e-09 | - |
| policy_mismatches | 0.00e+00 | 0e+00 | 0 of 760 states |
| stationary_residual | 1.96e-13 | 1e-10 | recurrent classes per case: 13, 1, 19 |
| mass_error | 2.22e-16 | 1e-12 | - |

- QuantEcon 0.13 has no household/Aiyagari class, so the same discrete-choice problem (state = assets x income, action = next-asset grid index) is built for `DiscreteDP` with an independently written CRRA utility and budget constraint.
- Inputs shared by construction: Dynecon's asset grid, income values and transition matrix. Cases: CRRA 2.0 / 1.5 / 3.0, r 0.02-0.03, beta 0.95-0.96, borrowing limit 0 and -1.
- Policy mismatches are counted over grid indices (0 means identical choices).
- On a discrete grid the optimal-policy chain has several recurrent classes, so its stationary distribution is not unique and cannot be compared point by point. The check is instead that Dynecon's distribution is invariant under QuantEcon's own controlled chain (mu P = mu) and sums to one.
- The EGM solver is continuous-choice, so it is not expected to match a discrete-grid solution exactly; its accuracy is covered by the Euler-error diagnostics.

# Performance and accuracy against other libraries

Generated from `crosscheck/results/libs_benchmark.json` by `scripts/gen_crosscheck_doc.py`;
regenerate the data with `scripts/benchmark_vs_libs.py`. Wall-clock seconds, minimum of 7 warm runs; first-call (JIT) times are in the JSON. Single machine (Windows-11-10.0.26200-SP0, 12 logical cores), Python 3.14.8, dynecon 0.1.1, quantecon 0.13.1, sequence-jacobian 1.0.0, numpy 2.5.4. Both libraries receive the same grid, income values and transition matrix; one machine and one problem family, so treat the ratios as indicative.

## Household EGM + stationary distribution vs Sequence-Space Jacobian (SSJ)

SSJ's `hh` block is also an endogenous-grid solver, so this compares like with like. Tolerances are matched (policy 1e-8, distribution 1e-10).

| assets | SSJ (s) | Dynecon Numba (s) | Dynecon NumPy (s) | Dynecon / SSJ (Numba) | policy diff | distribution diff |
|---|---|---|---|---|---|---|
| 200 | 0.040 | 0.036 | 0.112 | 0.9x | 1.2e-07 | 2.0e-09 |
| 500 | 0.078 | 0.067 | 0.131 | 0.9x | 1.1e-07 | 2.6e-09 |
| 1000 | 0.119 | 0.108 | 0.185 | 0.9x | 1.1e-07 | 2.5e-09 |

Dynecon is faster on this task (Dynecon/SSJ time ratio 0.86x to 0.90x). The two agree to 1e-07 in the policy and 3e-09 in the distribution, which is the size expected from iteration tolerances of 1e-8 and 1e-10.

## Discrete savings problem vs QuantEcon `DiscreteDP`

Same discrete-choice problem and grid (see the cross-check above), tolerance 1e-8, model construction excluded for QuantEcon. `DiscreteDP` is a generic solver that does not exploit the monotone structure of the savings policy.

| states | QuantEcon policy iter. (s) | QuantEcon value iter. (s) | Dynecon VFI Numba (s) | speed-up vs policy iter. | policy mismatches |
|---|---|---|---|---|---|
| 180 | 0.062 | 0.160 | 0.005 | 11.4x | 0 |
| 300 | 0.218 | 1.697 | 0.008 | 26.2x | 0 |
| 450 | 0.324 | 4.696 | 0.013 | 24.0x | 0 |

Not benchmarked: HARK (its income process and normalisation differ, so no identical problem has been set up yet) and econpizza/Dynare (different problem classes).
