# EconKit

A Python Library for Computational Economics.

## Current Implementation Status

The repository now includes the initial implementation foundation:

- Protocol-first core scaffolding.
- Custom backend dispatcher infrastructure.
- NumPy reference interpolation kernel.
- Optional Numba interpolation kernel.
- NumPy reference shock discretization (Tauchen/Rouwenhorst).
- NumPy reference quadrature/expectation kernels.
- NumPy reference bounded optimization (golden section + Brent-style).
- Minimal model DSL and immutable compile pipeline (Session 5 vertical slice).
- Solver foundation slice with VFI scaffold and Bellman residual diagnostics (Session 6).
- Session 7 solver extensions: Howard-style policy-evaluation hook and compiled-model VFI adapter.
- Session 8 additions: canonical consumption-savings VFI helper, residual-history reporting, and selected true Numba backend kernels.
- Session 9 additions: stochastic-income canonical helper, expanded Numba backend coverage, and JSON-ready convergence artifact persistence.
- Session 10 additions: liquid/illiquid two-asset prototype helper, policy monotonicity diagnostics, iteration hooks, and schema-versioned artifact export.
- Session 11 additions: first simulation vertical slice, schema validation helpers, and performance hardening knob for the liquid/illiquid helper.
- CI, tests, and contribution standards with economics-explainability requirements.

## Quick Start

```bash
python -m venv .econkit
source .econkit/bin/activate
pip install -e .[dev]
pytest
```

## Why the Dispatcher Matters

EconKit routes numerical operations through a custom dispatcher so callers use one API while runtime selects NumPy, Numba, and future JAX/C/Rust implementations.

## Project Plan

The execution plan is active and tracked in the architecture and session logs under `docs/md/`.

## Documentation Layout

- `docs/md/`: architecture notes, explainability standard, and captain's log.
- `docs/latex/`: formal mathematical formulas and derivations that map to code.

## Review Milestones

- `docs/md/REVIEW_1.md`: first formal review checkpoint agenda and criteria.
- `docs/md/BACKEND_ROADMAP.md`: backend parity priorities and implementation sequence.