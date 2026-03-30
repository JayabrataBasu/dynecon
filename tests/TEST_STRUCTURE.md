# Test Structure

Tests are grouped by domain to reduce developer confusion:

- `tests/infrastructure/`: dispatch, config, and runtime wiring behavior.
- `tests/numerics/`: numerical kernel correctness and invariants.
- `tests/solvers/`: solver algorithms, convergence behavior, and diagnostics.
- `tests/cross_backend/`: parity checks for equivalent APIs across backends.
- `tests/simulation/`: policy-path simulation correctness and summary statistics.

As the project grows, add new folders by domain (for example: `tests/solvers/`, `tests/simulation/`, `tests/estimation/`).
