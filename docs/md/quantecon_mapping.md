# Dynecon and QuantEcon.py: routine-by-routine mapping (draft)

**Status: draft, written before the feature freeze.** QuantEcon.py 0.13.1 was enumerated by
introspecting the installed package; Dynecon by reading its `__all__` lists. Statuses:

- **Equivalent**: same job, same inputs and outputs in spirit. Where a numerical cross-check exists it is named.
- **Partial**: covers some of the routine's functionality or a narrower case.
- **Absent**: Dynecon has nothing comparable.
- **Out of scope**: deliberately not planned (not macro or heterogeneous-agent estimation).

Only the cross-checks in [crosscheck](crosscheck.md) are evidence of numerical agreement; every other
"Equivalent" means "does the same job", not "verified against QuantEcon".

## Markov chains and discretisation (`quantecon.markov`)

| QuantEcon | Dynecon | Status | Note |
|---|---|---|---|
| `tauchen` | `numerics.tauchen` | Equivalent | cross-checked, 40 cases, grid and P to 1e-12 |
| `rouwenhorst` | `numerics.rouwenhorst` | Equivalent | cross-checked as above |
| `discrete_var` (multivariate AR) | none | Absent | |
| `std_norm_cdf` | scipy | Out of scope | |
| `MarkovChain` (class) | `numerics.MarkovChain` | Equivalent | cross-checked on 40 random chains: classes, recurrence, irreducibility, aperiodicity, stationary distributions |
| `MarkovChain.is_irreducible`, `communication_classes`, `recurrent_classes`, `is_aperiodic`, `period` | same properties on `MarkovChain` | Equivalent | `cyclic_classes` not implemented |
| `mc_compute_stationary` (one per recurrent class) | `MarkovChain.stationary_distributions` | Equivalent | |
| `gth_solve` | inside `stationary_distribution` | Equivalent | same GTH algorithm; cross-checked |
| `mc_sample_path`, `MarkovChain.simulate` | `numerics.simulate_markov`, `markov_paths` | Equivalent | |
| `estimate_mc`, `fit_discrete_mc` | none | Absent | |
| `random_markov_chain`, `random_stochastic_matrix` | none | Absent | |
| `DiGraph` and graph tools | none | Absent | |

## Dynamic programming (`quantecon.markov.ddp`)

| QuantEcon | Dynecon | Status | Note |
|---|---|---|---|
| `DiscreteDP` (value iteration, policy iteration, modified PI) | `FiniteDPProblem` + `solve(method="vfi")` | Partial | value iteration with Howard steps; no pure policy iteration, no linear programming |
| `DiscreteDP.solve` on a household grid | `solve(model, method="vfi")` | Equivalent | cross-checked, 3 cases, value to 1e-9, policy identical |
| `backward_induction` (finite horizon) | none | Absent | |
| `random_discrete_dp`, `sa_indices` | none | Absent | |
| `ddp_linprog_simplex` | none | Absent | |
| Dynecon-only: EGM for continuous choice | `solve(method="egm")` | Dynecon-only | with Anderson acceleration |
| Dynecon-only: stationary equilibrium solver | `solve_stationary_equilibrium` | Dynecon-only | Aiyagari, Huggett |

## Linear-quadratic control and linear systems

| QuantEcon | Dynecon | Status | Note |
|---|---|---|---|
| `LQ` (finite and infinite horizon) | `solvers.lq.LQ` | Equivalent | cross-checked on 20 random systems (discounting, noise, cross terms; both horizons) |
| `LQMarkov` | none | Absent | |
| `RBLQ` (robust control) | none | Absent | |
| `nnash` (LQ Nash game) | none | Absent | |
| `solve_discrete_riccati` | `solvers.lq.solve_discrete_riccati` | Equivalent | same values as `LQ`; `solve_discrete_riccati_system` (stacked) absent |
| `solve_discrete_lyapunov` | scipy, used inside `theoretical_moments` | Partial | not exposed |
| `LinearStateSpace` | `solvers.perturbation` simulate and moments | Partial | `simulate_linear`, `theoretical_moments`, `irf` cover the DSGE case |
| `LinearStateSpace.moment_sequence`, `stationary_distributions`, `geometric_sums` | `theoretical_moments` | Partial | covariances and autocovariances only |
| `Kalman` (filter) | `estimation.kalman_filter` | Equivalent | cross-checked, 20 random systems, 1e-10 |
| `Kalman` stationary values (`K_infinity`, `Sigma_infinity`) | internal steady-state shortcut | Partial | not exposed |
| `Kalman` smoother | none | Absent | |
| `DLE` (dynamic linear economies) | none | Absent | |
| `var_quadratic_sum`, `m_quadratic_sum` | none | Absent | |
| Dynecon-only: Klein linear rational-expectations solver | `klein_solve`, `solve_perturbation` | Dynecon-only | verified on Smets-Wouters (mode matches Dynare) |
| Dynecon-only: JAX Kalman log-likelihood | `kalman_loglik_jax` | Dynecon-only | |
| Dynecon-only: exact likelihood gradient of linear Dynare models | `dynare_grad.loglik_and_grad` | Dynecon-only | implicit differentiation of the solution; checked against finite differences |
| Dynecon-only: Dynare `model(linear)` reader | `dynare.parse_linear_model` | Dynecon-only | reproduces the Smets-Wouters posterior mode |

## Quadrature and grids (`quantecon.quad`, `gridtools`)

| QuantEcon | Dynecon | Status | Note |
|---|---|---|---|
| `qnwnorm`, `qnwlogn` | `numerics.mvnormal_nodes`, `lognormal_nodes` | Equivalent | multivariate product rules; integrals of test functions agree to 1e-9 |
| `qnwlege` | `numerics.gauss_legendre` | Partial | univariate; combine with `tensor_rule` |
| `qnwcheb`, `qnwtrap` | `numerics.gauss_chebyshev`, `trapezoid_rule` | Partial | different rule definitions from QuantEcon's; univariate |
| `qnwsimp`, `qnwunif`, `qnwbeta`, `qnwgamma`, `qnwequi`, `quadrect` | none | Absent | |
| `cartesian`, `gridmake` | `numerics.cartesian`, `tensor_rule` | Equivalent | `mlinspace`, `ckron` absent |
| `simplex_grid`, `cartesian_nearest_index` | none | Absent | |
| (none) | `exp_grid`, `uniform_grid`, `make_grid` | Dynecon-only | curved grids for borrowing constraints |
| `compute_fixed_point` | `numerics.fixed_point` | Equivalent | Dynecon adds Anderson acceleration |
| (none) | `linear_interp_1d`, `linear_interp_columns`, `numerical_jacobian` | Dynecon-only | |

## Optimisation and root finding (`quantecon.optimize`)

| QuantEcon | Dynecon | Status | Note |
|---|---|---|---|
| `brentq`, `bisect`, `newton`, `newton_halley`, `newton_secant` (Numba-compiled) | scipy `brentq` inside the equilibrium solver | Out of scope | Dynecon relies on SciPy, not exposed |
| `nelder_mead`, `brent_max` (Numba-compiled) | scipy | Out of scope | |
| `linprog_simplex`, `lcp_lemke`, `minmax` | none | Out of scope | game-theory and LP helpers |

## Time series, inequality and statistics

| QuantEcon | Dynecon | Status | Note |
|---|---|---|---|
| `ARMA` | none | Absent | |
| `periodogram`, `smooth`, `ar_periodogram` | none | Absent | |
| `hamilton_filter` | `data.hamilton_filter` | Equivalent | not yet cross-checked |
| (none) | `hp_filter`, `baxter_king` | Dynecon-only | |
| `lorenz_curve`, `gini_coefficient` | `simulation.lorenz_curve`, `gini` | Equivalent | weighted and exact; not yet cross-checked |
| `shorrocks_index`, `rank_size` | none | Absent | |
| `ECDF`, `DiscreteRV`, `BetaBinomial`, `LAE` | none | Absent | |
| `probvec`, `draw`, `sample_without_replacement` | none | Out of scope | |
| (none) | `autocorrelation`, `newey_west_variance`, `block_bootstrap_se`, `bootstrap_ci` | Dynecon-only | |

## Other QuantEcon modules

| QuantEcon | Dynecon | Status |
|---|---|---|
| `game_theory` (about 60 routines: Nash solvers, repeated games, best-response dynamics, generators) | none | Out of scope |
| `IVP` (initial value problems) | none | Out of scope |
| `rank_est`, `nullspace` | scipy | Out of scope |
| timing and notebook utilities | none | Out of scope |

## What Dynecon has that QuantEcon.py does not

| Area | Dynecon routines |
|---|---|
| Household problems with a continuous asset | EGM solver, Young-lottery distribution, `stationary_distribution_young` |
| Heterogeneous-agent equilibrium | `solve_stationary_equilibrium`, built-in Aiyagari and Huggett |
| Exact derivatives of stationary moments | `stationary_moments`, `equilibrium_moments`, `mixture_moments` (JAX) |
| Gradient-based estimation on those models | `smm_exact`, `quasi_posterior` |
| Estimation toolkit | `smm`, `mle`, `bayes_estimate`, `metropolis`, own `hmc`, `identification_report` |
| Linear DSGE solving and filtering | `klein_solve`, `solve_perturbation`, `irf`, `kalman_filter` |
| Policy experiments | `PolicyExperiment`, `cev`, `elasticities`, `ParameterSweep` |
| Reproducibility | provenance, model fingerprints, experiment records |
| Safety diagnostics | `GridWarning`, Euler errors, identification report |

## Summary of the mapping

| Status | Approximate count of QuantEcon routines |
|---|---|
| Equivalent | about 17 |
| Partial | about 6 |
| Absent, in scope for a macro library | about 45 |
| Out of scope (game theory, LP, timing, utilities) | about 100 |

Counts are approximate because QuantEcon's classes bundle many methods and the game-theory package
alone has about 60 routines.

## Reading the result

- **Where Dynecon replaces QuantEcon today:** discretisation, Markov simulation, the Kalman filter,
  discrete household problems, and fixed points.
- **Where a user still needs QuantEcon:** LQ control and Riccati equations, Markov chain structure
  (communication classes), multivariate quadrature, ARMA and spectral tools, and the Kalman smoother.
- **What neither covers:** aggregate shocks in heterogeneous-agent models, higher-order perturbation,
  occasionally binding constraints, continuous time.
- **Closed since the first draft:** Markov chain class, LQ and Riccati, multivariate normal and
  lognormal quadrature, `cartesian`.
- **Cheapest remaining gaps:** Kalman smoother, stacked Riccati system, `mlinspace`/`ckron`, ARMA and
  spectral tools.
