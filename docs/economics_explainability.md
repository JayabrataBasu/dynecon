# Economics Explainability Standard

EconKit code must explain economics, not just implementation details.

## Required in Model and Solver Code

- Every economic equation must have a docstring describing the economic object represented.
- Every objective function must state economic interpretation and units.
- Every solver loop must include a short note explaining the economic operation (for example, Bellman update or Euler inversion).
- Parameter fields should include interpretation and calibration intent where applicable.

## PR Review Checks

- Does the change preserve economic meaning in code comments/docstrings?
- Are assumptions and constraints stated explicitly?
- Is the numerical implementation traceable to an economics concept?

## Why This Matters

EconKit is a computational economics project. If a numerical routine cannot be connected back to the economics it represents, the implementation is incomplete.
