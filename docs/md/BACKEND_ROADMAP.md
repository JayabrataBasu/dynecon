# Backend roadmap

The Python surface API is permanent; compiled layers sit behind the dispatcher
(`dynecon/numerics/_dispatch.py`) and are added operation by operation, validated against the
NumPy reference, never replacing it. See `docs/md/Dynecon_Architecture.md`, Section 19.

## Status per operation

Regenerate with `python scripts/backend_matrix.py` (after importing the JAX kernels with
`dynecon.backends.describe()`); current state:

| operation | numpy | numba | jax |
|---|---|---|---|
| linear_interp_1d | yes | yes | yes |
| linear_interp_columns | yes | yes | yes |
| bellman_saving | yes | yes | yes |
| distribution_lottery | yes | yes | yes |
| markov_paths | yes | yes | yes |
| simulate_saving | yes | yes | - |

Cold-path operations (discretisation, quadrature, GTH stationary distribution, grid construction,
Jacobians, the Klein solver) have a single NumPy/SciPy implementation by design (rule R4);
perturbation Jacobians additionally have an exact JAX autodiff variant, and `solve_many`
vectorises EGM over parameters with `jax.vmap`.

## Layers

1. **NumPy reference**: always present, the correctness truth.
2. **Numba** (v1.0): hot loops on CPU; compiled lazily on first call, cached.
3. **JAX** (v1.0): autodiff, `vmap`, GPU. Not claimed to be faster on CPU.
4. **C via pybind11, Rust via PyO3**: after 1.0 only, and only for kernels where recorded
   benchmarks show the best existing backend missing its goal by a clear margin.

## Rules for any new layer

The NumPy reference stays; a kernel ships with a parity test and a benchmark case; the public API
does not change; the wheel matrix is documented before the first release.
