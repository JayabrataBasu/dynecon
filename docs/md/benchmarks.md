# Benchmarks

Generated from `20261103_JayabrataBasu.json` by `scripts/gen_bench_doc.py`. Timings are the **minimum** of the repeats; compile time is the first call minus the steady-state median. All numbers come from one machine; never compare across machines.

- Machine: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel, 12 logical cores, RAM None GB, Windows-11-10.0.26200-SP0
- Versions: dynecon 0.1.0.dev0, numpy 2.5.4, scipy 1.18.1, numba 0.68.1, jax 0.12.0.1, python 3.14.8
- Date: 2026-11-03

| case | backend | min (ms) | speedup vs NumPy | compile (s) | max abs diff vs NumPy |
|---|---|---:|---:|---:|---:|
| interp1d | numpy | 331.99 |  | 0.00 |  |
| interp1d | numba | 129.54 | 2.56x | 1.82 | 0.0e+00 |
| interp1d | jax | 154.09 | 2.15x | 2.82 | 1.1e-16 |
| bellman_cs | numpy | 119.36 |  | 0.00 |  |
| bellman_cs | numba | 3.94 | 30.31x | 0.08 | 0.0e+00 |
| bellman_cs | jax | 68.05 | 1.75x | 0.34 | 0.0e+00 |
| vfi_cs | numpy | 2015.92 |  | 0.14 |  |
| vfi_cs | numba | 260.33 | 7.74x | 0.03 | 0.0e+00 |
| egm_cs | numpy | 1097.79 |  | 0.00 |  |
| egm_cs | numba | 792.73 | 1.38x | 0.11 | 0.0e+00 |
| egm_cs | jax | 1820.17 | 0.60x | 0.78 | 4.4e-16 |
| egm_rbc | numpy | 533.27 |  | 0.00 |  |
| egm_rbc | numba | 328.68 | 1.62x | 0.00 | 0.0e+00 |
| young_stationary | numpy | 131.60 |  | 0.00 |  |
| young_stationary | numba | 31.37 | 4.19x | 0.04 | 4.3e-19 |
| young_stationary | jax | 373.77 | 0.35x | 0.28 | 8.7e-19 |
| simulate_panel | numpy | 11029.43 |  | 0.00 |  |
| simulate_panel | numba | 551.07 | 20.01x | 0.03 | 0.0e+00 |
| aiyagari_ge | numpy | 3447.27 |  | 0.00 |  |
| perturbation_rbc | numpy | 60.51 |  | 0.03 |  |
| perturbation_rbc | jax | 135.98 | 0.44x | 3.03 | 3.5e-10 |
| kalman_loglik | numpy | 137.89 |  | 0.00 |  |
| smm_ar1 | numpy | 2467.92 |  | 0.03 |  |

## Goals versus measurements

Goals are never edited to fit results.

- `bellman_cs` (Numba >= 20x NumPy vectorised): measured 30.31x, goal 20x: met
- `simulate_panel` (Numba >= 10x NumPy loop): measured 20.01x, goal 10x: met
- `young_stationary` (Numba >= 3x NumPy histogram step): measured 4.19x, goal 3x: met
- `vfi_cs` (full solve, Numba monotone + Howard vs vectorised NumPy): 7.74x, goal 5x: met
- `egm_cs` (EGM at n_a=1000 vs Numba VFI at n_a=500): 0.33x faster, goal 20x: **missed**

**Why EGM misses its goal.** EGM contracts at rate beta, so it needs about 360 iterations to reach its default 1e-10 tolerance, each costing roughly 2 ms of array arithmetic (`cProfile`: the CRRA power functions on the 7000-element arrays and the interpolation call dominate after the batched `linear_interp_columns` fix removed per-call validation, which cut the solve from about 1.2 s to about 0.8 s). The Numba VFI with Howard acceleration needs far fewer outer iterations. EGM's advantage in this library is accuracy (Euler errors orders of magnitude smaller than VFI at any grid size), not speed. A follow-up is parked in `V1_PROGRESS.md`.

### Legacy pure-Python VFI (git tag `v0.1.0-baseline`)

One Bellman iteration at (n_a, n_z) = (200, 7): legacy 0.186 s (measured over 2 iterations in a temporary worktree), Numba kernel 0.093 ms: **1992x** per iteration (goal 50x for the full solve; Howard acceleration and monotone search reduce the iteration count further).

## Notes

- `simulate_panel` uses 2x10^4 agents (not 10^5) to keep peak memory modest.
- JAX is not claimed to be faster on CPU; it exists for autodiff, `vmap` and GPU.
- Timing is never a required CI check.
