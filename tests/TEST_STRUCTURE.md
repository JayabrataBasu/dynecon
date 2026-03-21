# Test Structure

Tests are grouped by domain to reduce developer confusion:

- `tests/infrastructure/`: dispatch, config, and runtime wiring behavior.
- `tests/numerics/`: numerical kernel correctness and invariants.

As the project grows, add new folders by domain (for example: `tests/solvers/`, `tests/simulation/`, `tests/estimation/`).
