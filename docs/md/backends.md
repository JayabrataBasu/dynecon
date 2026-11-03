# Backends: when to use NumPy, Numba and JAX

Every hot numerical operation is registered with the dispatcher under an `(operation, backend)`
key. NumPy is the permanent reference and the universal fallback; Numba and JAX add
implementations of the same operations, tested against NumPy for agreement.

```python
import dynecon as ek

ek.config.set_backend("numba")  # validated; "auto" (default) = Numba if installed
ek.backends.available_backends()  # ['numpy', 'numba', 'jax']
print(ek.backends.describe())  # operation x backend table, versions, JAX device
sol = ek.solve(model, backend="numpy")  # per-call override
```

| Backend | Use it for | Notes |
|---|---|---|
| `numpy` | correctness, debugging, small problems, no optional dependencies | the reference every other backend is checked against |
| `numba` | speed on CPU for tight loops (Bellman maximisation, simulation, histogram step) | compiles on first call (`cache=True`); `DYNECON_DISABLE_JIT=1` runs the same code in pure Python |
| `jax` | exact derivatives (perturbation Jacobians, likelihood gradients), `vmap` over parameters (`solve_many`), GPU | enables 64-bit precision globally on first use; often **slower** than Numba on CPU |

## Resolution rule

The requested backend is used if it implements the operation; otherwise Dynecon falls back to
NumPy, or raises `BackendError` when `config.strict_backend = True`. Requesting an uninstalled
backend raises `BackendUnavailableError` in strict mode and otherwise warns once and uses NumPy.
The backend actually used is recorded on `Solution.backend`.

## Determinism

Random numbers are drawn in NumPy before compiled kernels run, so results do not depend on the
number of threads. Different backends agree to rounding (about 1e-12) for the same inputs.

## Honest performance

Speedups are measured, not claimed: see [benchmarks.md](benchmarks.md) (regenerated from the
recorded JSON by `scripts/gen_bench_doc.py`). Compile time is reported separately from
steady-state time.
