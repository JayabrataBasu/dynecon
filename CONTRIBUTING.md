# Contributing to EconKit

Thank you for contributing.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Contribution Principles

- Economics first: code and docs must explain the economics being represented.
- Correctness first: NumPy reference behavior is the baseline.
- Speed matters: performance-critical changes should include benchmark notes.

## Required Checks

- `ruff check .`
- `mypy econkit`
- `pytest`

## PR Expectations

Use the PR template and provide:

- Economic interpretation of changes.
- Numerical implications and risk.
- Performance impact and evidence when relevant.
