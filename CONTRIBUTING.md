# Contributing to Dynecon

Thank you for contributing.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install                        # pip install -e ".[dev]"
make ci                             # all gates
```

## Principles

- **Economics first.** Code and docs must explain the economics: public symbols carry an
  `Economics note:` line (own docstring or module docstring; `scripts/check_docstrings.py`).
- **Reference first.** Every hot numerical kernel starts as a clear NumPy reference with tests; a
  Numba/JAX implementation is added afterwards, registered under the same operation name, and checked
  against the reference by a parity test.
- **Analytic benchmarks.** Prefer tests against closed forms (Brock-Mirman, permanent income, ...) and
  property-based tests (Hypothesis). Deterministic seeds, no network, no wall-clock assertions.
- **Lean by construction (R21).** One concept, one implementation. Every public symbol has a caller
  and a test (`make deadcode`). No compatibility shims before 1.0, no speculative generality. Code that
  will be used by a concrete upcoming step may carry `# NEAR-FUTURE(P<n>.<m>): ...`.
- **Honest numbers.** Performance claims come only from `docs/md/benchmarks.md`, generated from
  recorded runs (`python -m benchmarks.run --all`, then `python scripts/gen_bench_doc.py`).

## Required checks

```bash
make ci        # test-structure + progress check, ruff, mypy, pytest, vulture, usage/near-future checks
make coverage  # core/numerics/model/solvers >= 90 %, others >= 80 %
```

Tests live in `tests/<domain>/` (see `tests/TEST_STRUCTURE.md`). Markers: `slow`, `numba`, `jax`,
`network`, `plot`. CI also runs without Numba and with JAX.

## Adding a model

Write `dynecon/models/<name>.py` with `PARAMS`, `DEFAULTS`, `CALIBRATION_SOURCE`, `ANALYTIC`, a
`build(**overrides)` function returning a `Model`, and end with `register("<name>", build)`. The
registry's `register(name, builder)` is the same function the nine built-ins use; there is no separate
plugin mechanism. Add a test against an analytic or published benchmark, an entry in
`scripts/gen_model_docs.py`, and a CHANGELOG line.

## Commits and PRs

Small, reviewable commits (`P<phase>.<step>: summary` for plan work). Use the PR template and
describe the economic interpretation of the change. Do not push tags or publish packages: releases are
human-gated.
