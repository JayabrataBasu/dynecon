# What "1.0" means for Dynecon

Set by the maintainer on 2026-11-04. 1.0 is a more complete release, not a relabelled 0.1.

1. **Mathematically sound.** Every solver is checked against analytic or published benchmarks
   (Brock-Mirman, Aiyagari, Lucas tree, RBC, NK) and cross-checked against an independent
   implementation where one exists (open item P10.13, QuantEcon). No known incorrect results.
2. **Usage and edge cases covered.** Bad inputs, degenerate parameters, non-convergence,
   indeterminacy, extreme grids and missing optional dependencies all give clear errors or
   documented behaviour. Property-based tests for the numerical core. Minimum dependency
   versions are tested in CI.
3. **No known vulnerabilities.** `pip-audit` clean, Actions pinned, trusted publishing.
4. **More features than the alternatives.** A differentiating capability beyond today's set
   (candidates: HANK-style aggregate shocks, sequence-space Jacobians, differentiable estimation
   end to end with JAX, richer models). Chosen and scoped before the release candidate.
5. **Plan gates closed.** Review-2/3/4, a release candidate, and a soak period
   (see `V1_EXECUTION_PLAN.md`).

Open decision: which headline feature(s) satisfy item 4.
