# Experiments and reproducibility

## Reproducible runs

```python
from econkit.experiment import Experiment

exp = Experiment(
    "aiyagari_base",
    build=("consumption_savings", {"beta": 0.95}),
    solve_options={"method": "egm"},
    simulate_options={"n_periods": 500, "n_agents": 200},
    moments=["mean_assets", "std_consumption"],
    outdir="runs",
)
result = exp.run(rng=123)
report = Experiment.reproduce(result.path)  # re-runs and compares numerically
assert report.passed
```

A run folder `runs/<name>/<timestamp>-<fingerprint8>/` holds `config.json`, `provenance.json`
(package versions, git commit, platform, model fingerprint), `solution.npz` + `solution.json`,
`result.json` and `log.txt`. Models are never serialised as code: `build` names how to rebuild
them (a built-in model plus overrides, or an importable `"module:function"`).

**The contract.** Reproduction is guaranteed on the same machine, library versions, backend and
thread count, with relative tolerance 1e-10. Across machines or versions results should agree
to the documented numerical tolerances but are not bit-identical.

## Sweeps and elasticities

`ParameterSweep(model, grid, function)` solves over a Cartesian grid (or list of points) and
evaluates `function(solution)`; failed points are recorded, not fatal. `n_jobs > 1` uses a
process pool and needs a picklable build spec and a module-level function.
`backend="jax", method="egm"` solves the grid in one vectorised batch. `elasticities` computes
central log-differences `d ln metric / d ln param`.

## Policy experiments and welfare

`PolicyExperiment(baseline, reform)` solves both economies; a reform is a dict of parameter
overrides or a function `Model -> Model` (`Model.copy`, `replace_state`).

Welfare is the **consumption-equivalent variation** `lambda`: the proportional change in
baseline consumption, in all periods and states, that makes the household as well off as under
the reform. CRRA with `sigma != 1`: `lambda = (V_reform / V_base)^(1/(1-sigma)) - 1`; log
utility: `lambda = exp((1-beta)(V_reform - V_base)) - 1`. EGM solutions carry no value function,
so `evaluate_policy_value` computes `V = (I - beta T)^-1 u` from the policy.

The aggregate `lambda` evaluates both value functions on the **baseline stationary
distribution** (reform values are interpolated onto the baseline grid), i.e. the gain for the
households alive when the reform is introduced. Transition costs are out of scope for v1.

## Resampling

`monte_carlo(fn, n_reps, rng)` uses independent spawned streams. `bootstrap_ci` gives percentile
intervals; pass `block_len` for dependent data (moving-block bootstrap), otherwise standard
errors are too small for persistent series.
