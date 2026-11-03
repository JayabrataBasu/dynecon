"""Smoke test an *installed* dynecon (run from outside the repository, e.g. in a clean venv)."""

from __future__ import annotations

import sys

import numpy as np

import dynecon as ek
from dynecon import models


def main() -> int:
    print("dynecon", ek.__version__, "from", ek.__file__)
    if "site-packages" not in ek.__file__:
        print("warning: not running from site-packages")
    sol = ek.solve(models.load("consumption_savings", n_a=60, n_z=3).compile())
    assert sol.converged and sol.method == "egm"
    sim = ek.simulate(sol, n_periods=100, n_agents=10, rng=0)
    assert np.isfinite(sim.mean("assets"))
    rbc = ek.solve(models.load("rbc").compile())
    assert rbc.method == "perturbation"
    eq = models.huggett.equilibrium(n_a=40, n_z=3)
    assert eq.converged and abs(eq.aggregates["bond_holdings"]) < 1e-6
    for heavy in ("pandas", "matplotlib", "jax"):
        assert heavy not in sys.modules, f"{heavy} imported unexpectedly"
    print(ek.backends.describe())
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
