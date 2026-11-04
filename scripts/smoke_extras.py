"""Smoke test the optional stacks of an *installed* dynecon (run outside the repository).

Exercises JAX, pandas and matplotlib; each section is skipped (and reported) if the library is
not installed. Run ``smoke_installed.py`` first for the lazy-import checks.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np

import dynecon as ek
from dynecon import models


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_jax() -> None:
    model = models.load("consumption_savings", n_a=60, n_z=3).compile()
    ref = ek.solve(model, backend="numpy")
    jx = ek.solve(model, backend="jax")
    assert jx.converged
    assert jx.backend == "jax" and set(jx.policy) == set(ref.policy)
    for name in ref.policy:
        np.testing.assert_allclose(jx.policy[name], ref.policy[name], atol=1e-8)
    print("jax: egm parity with numpy ok;", ek.backends.describe().splitlines()[-1])


def check_pandas() -> None:
    from dynecon.data import hp_filter, parse_fred_csv

    s = parse_fred_csv(
        "observation_date,X\n2000-01-01,1.0\n2000-04-01,2.0\n2000-07-01,.\n2000-10-01,4.0\n"
    )
    assert len(s) == 4 and int(s.isna().sum()) == 1
    cycle_trend = hp_filter(np.sin(np.linspace(0, 6, 80)) + np.linspace(0, 3, 80))
    assert cycle_trend is not None
    print("pandas: parse_fred_csv and hp_filter ok")


def check_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from dynecon.simulation import marginal_distribution, stationary_distribution_young
    from dynecon.viz import plot_distribution, plot_lorenz

    sol = ek.solve(models.load("consumption_savings", n_a=60, n_z=3).compile())
    mass = marginal_distribution(stationary_distribution_young(sol), 0)
    fig = plot_distribution(mass, sol.grids["assets"])
    ax = plot_lorenz(sol.grids["assets"], mass)
    assert len(fig.axes) == 2 and "Gini" in ax.get_legend().get_texts()[0].get_text()
    print("matplotlib: plot_distribution and plot_lorenz ok")


def main() -> int:
    ran = 0
    for lib, check in (
        ("jax", check_jax),
        ("pandas", check_pandas),
        ("matplotlib", check_matplotlib),
    ):
        if _has(lib):
            check()
            ran += 1
        else:
            print(f"{lib}: not installed, skipped")
    print(f"extras smoke test passed ({ran} of 3 sections ran)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
