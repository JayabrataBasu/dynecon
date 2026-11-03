# Simulation and distributions

Two ways to describe a cross-section of households solving a `saving` problem:

| Method | Use it for | Cost |
|---|---|---|
| `simulate()` (particles) | time series, panels, non-stationary experiments, SMM moments | noisy; scales with agents x periods |
| Young (2010) histogram | stationary-equilibrium objects (aggregates, wealth distribution, Gini) | noise-free; scales with grid size |

## Panel simulation

```python
sim = ek.simulate(sol, n_periods=500, n_agents=1000, burn_in=100, rng=42)
sim.mean("assets")
sim.moments(["mean_assets", "std_consumption", "autocorr_income"])
sim.to_dataframe()  # long format with `period`, `agent` (needs pandas)
```

All shock indices are drawn **up front** in NumPy; a deterministic kernel then evaluates the
policy by linear interpolation in the continuous state (clamped at the grid ends). Results do
not depend on the number of threads and agree between backends to rounding. `store=[...]`
limits the stored variables; `shocks=` accepts index paths for common random numbers.

## Young histograms

`stationary_distribution_young(sol)` returns the mass `mu[i, j]` on the (asset, shock) grid.
For a policy value `a'` the mass is split between the two neighbouring nodes with weights that
preserve the policy mean exactly, then mixed across shocks with the transition matrix.
Iteration uses the shared `fixed_point` driver; if it has not converged within `max_iter`
(slowly mixing wealth near `beta (1 + r) = 1`) a sparse direct solve of `mu = T' mu` is used.
Mass outside the grid is clamped; check the mass at the top node when choosing `a_max`.

## Inequality and aggregates

`aggregate`, `marginal_distribution`, `cdf`, `quantile`, `lorenz_curve`, `gini` work on the
histogram or on any weighted sample. Wealth with a negative borrowing limit must be shifted
before computing Lorenz/Gini.

## Impulse responses

`simulate_irf(sol, shock_steps=1, ...)` computes generalised IRFs by paired simulations with
common random numbers (mean of shocked minus baseline). The shock moves the discrete shock
state by `shock_steps` grid points; for log-AR(1) shocks that is a known multiple of the
innovation standard deviation. For DSGE solutions use `dynecon.solvers.perturbation.irf`.
