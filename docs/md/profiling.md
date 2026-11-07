# Profiling the slowest benchmark cases

Issue #8 (plan step P15.5). `docs/md/benchmarks.md` lists timings; this page records **where the time goes** in the slowest cases.
Method: the repository's own cases in `benchmarks/cases.py`, one warm-up call, then one run under `cProfile` (Python 3.12, NumPy and Numba backends as named,
one Windows 11 machine, nothing else running). Profiling adds overhead, so shares below are shares of the profiled run, and the unprofiled wall time is given next to each case.
Nothing was changed in the library as a result of this page; candidate changes are listed as such. EGM was profiled earlier and is not repeated here.

## `smm_ar1` (NumPy, 0.50 s unprofiled)

| Where | Calls | Cumulative | Share |
|---|---:|---:|---:|
| the user-supplied moment function (the benchmark's own `moments`) | 239 | 0.637 s | 91% |
| of which `scipy.signal.lfilter` | 239 | 0.114 s | 16% |
| of which `numpy.corrcoef` | 239 | 0.240 s | 34% |
| Dynecon's own `moments_at` wrapper (own time) | 239 | 0.004 s | under 1% |

The time is the cost of 239 evaluations of the moment function (100,000 simulated draws each) requested by the Nelder-Mead optimiser. The library's overhead is negligible.
Candidate change (not made): fewer objective evaluations, for example the gradient-based path `smm_exact` for models that support it, not faster library code.

## `aiyagari_ge` (NumPy, 0.52 s unprofiled)

| Where | Calls | Cumulative | Share |
|---|---:|---:|---:|
| `fixed_point` (28 calls during the price search) | 28 | 0.617 s | 90% |
| EGM operator (household) | 1,351 | 0.186 s | 27% |
| Young distribution step | 14,049 | 0.136 s | 20% |
| convergence distance in `fixed_point` | 15,400 | 0.096 s | 14% |
| `jit` dispatch wrapper | 15,470 | 0.084 s (own time) | 12% |
| least squares inside Anderson acceleration | 807 | 0.060 s | 9% |

Shares overlap (a call is counted in its parents). The distribution iteration runs about ten times more steps than the household iteration (14,049 against 1,351), each step on a tiny
100 x 5 array, so the per-iteration Python cost of `fixed_point` (0.617 s over about 15,400 iterations, roughly 40 microseconds: dispatch wrapper, distance, array conversion) is paid very often. Candidate changes (not made): the Young distribution is iterated with plain `fixed_point` (no acceleration; a direct sparse solve already exists in the code as a fallback when the iteration does not converge within `max_iter`), so accelerating it or using the direct solve more often would cut the step count; compiling the loop body would remove the per-step overhead. Expected size of the gain is not measured.

## `simulate_panel` (20,000 agents x 1,000 periods)

NumPy reference path (2.79 s unprofiled): `simulate_saving_numpy` is 77% of the profiled run (2.35 s). Inside it, 7,000 calls to the linear interpolation (one per period and income state) take 1.41 s,
of which `searchsorted` is 0.80 s; building the Markov paths takes 0.61 s (20%). This path is the NumPy reference implementation, kept for correctness checks; the Numba path is 16x faster (see `benchmarks.md`).

Numba path (0.17 s unprofiled): the compiled kernel is 0.096 s (54%); drawing the shock paths 0.049 s (28%); array handling in `simulate` itself 0.028 s (16%). The work is spread out; there is no single hotspot to remove.

## Reading

None of the three slow cases points to an implementation defect. `smm_ar1` is dominated by the user's moment function and the optimiser's evaluation count; `simulate_panel` on NumPy is a reference path by design;
the one case with a clear structural cost is the stationary-distribution loop inside `aiyagari_ge`, which is the item to work on if the equilibrium solve needs to be faster.
