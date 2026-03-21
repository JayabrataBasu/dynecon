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