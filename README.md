# EconKit

A Python Library for Computational Economics.

## Current Implementation Status

The repository now includes the initial implementation foundation:

- Protocol-first core scaffolding.
- Custom backend dispatcher infrastructure.
- NumPy reference interpolation kernel.
- Optional Numba interpolation kernel.
- CI, tests, and contribution standards with economics-explainability requirements.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Why the Dispatcher Matters

EconKit routes numerical operations through a custom dispatcher so callers use one API while runtime selects NumPy, Numba, and future JAX/C/Rust implementations.

## Project Plan

See `.github/prompts/plan-econKit.prompt.md` for the active implementation blueprint.
