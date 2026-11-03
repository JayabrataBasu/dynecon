# User guide

## Installation

```bash
pip install dynecon                 # NumPy + SciPy only
pip install "dynecon[numba]"        # compiled kernels on CPU (recommended)
pip install "dynecon[jax]"          # exact derivatives, vmap, GPU
pip install "dynecon[data,viz]"     # pandas and matplotlib
pip install "dynecon[all]"          # everything optional
```

| Extra | Adds | Needed for |
|---|---|---|
| `numba` | Numba | speed on CPU; the library runs without it |
| `jax` | JAX | `backend="jax"`, `solve_many`, gradient-based `mle` |
| `data` | pandas | `to_dataframe`, FRED parsing, `Series`-aware filters |
| `viz` | matplotlib | `dynecon.viz` plots |
| `docs` | Sphinx stack | building this site |
| `dev` | test and lint tools | contributing |

`import dynecon` loads none of the optional stacks.

## The mental model

```text
Model (builder) ──compile()──► CompiledModel ──solve()──► Solution ──simulate()──► SimulationResult
                                    │                          │                         │
                              with_params()           policy_function()          moments ─► smm() / mle() / bayes
```

1. **Declare** parameters, states, controls and a problem template (`saving`, `finite_dp`, `dsge`,
   `pricing`) with a `Model`, or `models.load("name")` for a built-in. See [the DSL](../md/model_dsl.md).
2. **Compile** to an immutable, validated `CompiledModel` (all specification errors are reported
   together; `compiled.fingerprint()` identifies the exact model).
3. **Solve** with `ek.solve(model, method="auto")`. `auto` picks EGM when the model supplies marginal
   utility, its inverse and the marginal return, VFI otherwise, perturbation for `dsge`.
4. **Simulate** policies, or compute exact stationary distributions with Young's method.
5. **Estimate** by matching simulated to data moments, by likelihood, or by MCMC.

## Conventions

- **Randomness.** Every stochastic function takes `rng` (a seed, a `numpy.random.Generator` or `None`).
- **Errors.** Everything derives from `dynecon.errors.DyneconError`; validation problems are also
  `ValueError`. Non-convergence warns (`ConvergenceWarning`) and sets `converged=False`; set
  `ek.config.strict_convergence = True` to raise instead.
- **Results.** `Solution` carries `policy`, `value`, `diagnostics`, `metadata` (versions and the
  model fingerprint) and `.save()` / `.load()`.
- **Units of dsge solutions.** Deviations from steady state, in log points for `log_variables`.

## Choosing a solver

| Situation | Method |
|---|---|
| Concave consumption-savings / growth problem, accurate policies | `egm` (default) |
| Non-concave or no marginal conditions | `vfi` (grid-exact, `O(spacing)` accuracy) |
| Discrete choice with finite states | `vfi` on a `finite_dp` problem |
| Linear dynamics around a steady state, estimation of DSGE | `perturbation` |
| Asset pricing in an endowment economy | `pricing` |
| Heterogeneous-agent equilibrium | `solve_stationary_equilibrium` |

## Performance tips

Install Numba; keep `n_a^2 n_z` modest for VFI (the payoff tensor) or use EGM; use `solve_many`
with `backend="jax"` for many parameter values; read [backends](../md/backends.md) and the recorded
[benchmarks](../md/benchmarks.md) before assuming a speedup.

## FAQ

**Do I have to write Numba code?** No. Your callables are NumPy functions; kernels only see arrays.
**Why does `import dynecon` not import Numba?** Kernels compile lazily on first use.
**Results differ slightly between backends.** They agree to about 1e-12; compare against NumPy.
**EGM raised "endogenous grid is not monotone".** The problem is non-concave; use `method="vfi"`.
