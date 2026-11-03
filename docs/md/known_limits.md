# Known limits (v1)

Honest list of what EconKit v1 does not do or does only approximately.

- **One continuous state.** `saving` problems have a single endogenous asset plus discrete or
  discretised shocks. Multi-asset (HANK-style) and aggregate-risk (Krusell-Smith) models are
  not in v1.
- **VFI accuracy** is O(grid spacing) (no continuous-choice refinement); EGM is the accurate
  method and is the default whenever the marginal fields are supplied.
- **EGM needs concavity.** A non-monotone endogenous grid raises an error instead of falling
  back; use VFI.
- **VFI memory.** The payoff tensor is `n_a^2 n_z` doubles (guarded by `max_memory_gb`).
- **Perturbation is first order only**, around the deterministic steady state; unit roots are
  treated as unstable.
- **Stationary comparisons only.** No transition paths, MIT shocks or aggregate fluctuations in
  heterogeneous-agent models.
- **Policy values outside the asset grid are clamped** in simulation and histograms; check
  `mass_at_top` / `n_clamped` when choosing the grid.
- **SMM** is derivative-free; with few simulated periods the objective is noisy even with common
  random numbers. Standard errors rely on asymptotics and on a simulation-based `Omega`.
- **JAX** requires JAX-traceable equations for autodiff (use `econkit.models.utility.log/exp`) and
  enables float64 globally; it is not faster than Numba on CPU.
- **FRED download** uses the public CSV export and has not been verified against the live
  service from this code base; tests never use the network.
- **Reproducibility** is guaranteed only for the same machine, versions, backend and threads.
- **Not in v1:** time iteration, projection methods, continuous-time models, second-order
  perturbation, HMC, SymPy front end, plotly/dashboards, C/Rust layers (see the plan, Part E).
