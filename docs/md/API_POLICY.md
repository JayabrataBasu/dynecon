# API policy

Dynecon follows [semantic versioning](https://semver.org). This page defines what is public.

## What is public

- Every name in the top-level `dynecon.__all__`:
  `Model, Param, State, Control, AR1Shock, Solution, solve, simulate, smm, config, backends,
  errors, __version__`.
- Every name in the `__all__` of these documented sub-namespaces:
  `dynecon.model`, `dynecon.models`, `dynecon.numerics`, `dynecon.simulation`,
  `dynecon.estimation`, `dynecon.experiment`, `dynecon.data`, `dynecon.viz`,
  `dynecon.solvers.perturbation`, `dynecon.solvers.equilibrium`, `dynecon.solvers.batch`,
  `dynecon.core.errors`.
- The documented keyword options of `solve()` per method (`VFIConfig`, `EGMConfig`,
  `PerturbationConfig`, `PricingConfig` field names and defaults). Config classes themselves
  are implementation details.

## What is private

Everything under `dynecon.numerics.backends`, anything beginning with an underscore, and the
compiled kernels. **Compiled layers are implementation details exempt from the API promise**:
upgrading may make code faster or choose a different engine, but results agree with the NumPy
reference within the documented tolerances.

## Stability promise

From 1.0.0, code written against the public API keeps running through 1.x. Breaking changes
need a major release and a deprecation window of at least two minor versions. Before 1.0 there
are no external users, so the API changes freely and old paths are deleted in the same commit
as their replacement (see `CHANGELOG.md`).

## Numerical results

Bug fixes that change numerical output (for example a more accurate algorithm) are minor
releases and are listed under *Changed* in the changelog. Golden-file tests in `tests/golden/`
guard silent drift.
