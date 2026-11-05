# Benchmarks

Generated from `20261105_JayabrataBasu.json` by `scripts/gen_bench_doc.py`. Timings are the **minimum** of the repeats; compile time is the first call minus the steady-state median. All numbers come from one machine; never compare across machines.

- Machine: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel, 12 logical cores, RAM 16.8 GB, Windows-11-10.0.26200-SP0
- Versions: dynecon 0.2.0, numpy 2.5.4, scipy 1.18.1, numba 0.68.1, jax 0.12.0.1, python 3.14.8
- Date: 2026-11-05

| case | backend | min (ms) | speedup vs NumPy | compile (s) | max abs diff vs NumPy |
|---|---|---:|---:|---:|---:|
| interp1d | numpy | 87.99 |  | 0.00 |  |
| interp1d | numba | 41.83 | 2.10x | 0.55 | 0.0e+00 |
| interp1d | jax | 79.19 | 1.11x | 1.08 | 1.1e-16 |
| bellman_cs | numpy | 39.88 |  | 0.00 |  |
| bellman_cs | numba | 1.95 | 20.43x | 0.03 | 0.0e+00 |
| bellman_cs | jax | 23.40 | 1.70x | 0.11 | 0.0e+00 |
| vfi_cs | numpy | 628.52 |  | 0.09 |  |
| vfi_cs | numba | 65.76 | 9.56x | 0.01 | 0.0e+00 |
| egm_cs | numpy | 204.86 |  | 0.00 |  |
| egm_cs | numba | 124.76 | 1.64x | 0.02 | 0.0e+00 |
| egm_cs | jax | 319.99 | 0.64x | 0.13 | 1.6e-13 |
| egm_cs_500 | numba | 70.21 |  | 0.01 |  |
| egm_rbc | numpy | 69.08 |  | 0.00 |  |
| egm_rbc | numba | 35.96 | 1.92x | 0.03 | 0.0e+00 |
| young_stationary | numpy | 30.61 |  | 0.00 |  |
| young_stationary | numba | 8.89 | 3.44x | 0.01 | 4.3e-19 |
| young_stationary | jax | 59.57 | 0.51x | 0.13 | 4.3e-19 |
| simulate_panel | numpy | 3236.74 |  | 0.00 |  |
| simulate_panel | numba | 192.60 | 16.81x | 0.01 | 0.0e+00 |
| aiyagari_ge | numpy | 568.51 |  | 0.04 |  |
| perturbation_rbc | numpy | 2.38 |  | 0.01 |  |
| perturbation_rbc | jax | 29.25 | 0.08x | 1.22 | 3.5e-10 |
| kalman_loglik | numpy | 8.05 |  | 0.00 |  |
| smm_ar1 | numpy | 787.20 |  | 0.00 |  |

## Goals versus measurements

Goals are never edited to fit results.

- `bellman_cs` (Numba >= 20x NumPy vectorised): measured 20.43x, goal 20x: met
- `simulate_panel` (Numba >= 10x NumPy loop): measured 16.81x, goal 10x: met
- `young_stationary` (Numba >= 3x NumPy histogram step): measured 3.44x, goal 3x: met
- `vfi_cs` (full solve, Numba monotone + Howard vs vectorised NumPy): 9.56x, goal 5x: met
- `egm_cs` (EGM at n_a=1000 vs Numba VFI at n_a=500): 0.53x faster, goal 20x: **missed**
- `egm_cs_500` (EGM vs Numba VFI on the same grid, n_a=500): EGM is 0.94x as fast as VFI (above 1 means EGM is faster)

**Why EGM misses its goal.** The goal compares EGM on a grid twice as large with VFI at n_a=500 and asks for 20x. EGM's work grows roughly linearly with the grid, and it is Anderson-accelerated since 0.2.0 (about half the iterations); the Numba VFI with Howard acceleration and monotone search needs few outer iterations on small grids. EGM's advantage in this library is accuracy and scaling to large grids, not a 20x speed-up. The numbers above are the recorded ones; the goal stays as written.

### Legacy pure-Python VFI (git tag `v0.1.0-baseline`)

One Bellman iteration at (n_a, n_z) = (200, 7): legacy 0.248 s (measured over 2 iterations in a temporary worktree), Numba kernel 0.076 ms: **3276x** per iteration (goal 50x for the full solve; Howard acceleration and monotone search reduce the iteration count further).

## Notes

- `simulate_panel` uses 2x10^4 agents (not 10^5) to keep peak memory modest.
- JAX is not claimed to be faster on CPU; it exists for autodiff, `vmap` and GPU.
- Timing is never a required CI check.
