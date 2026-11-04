"""Compare Dynecon with QuantEcon on identical problems; write ``crosscheck/results/quantecon.json``.

Requires the ``crosscheck`` extra (``pip install -e ".[crosscheck]"``). Each section records the
number of cases, the largest absolute differences and a verdict against a stated tolerance, so
``scripts/gen_crosscheck_doc.py`` can render the report without recomputing anything.
"""

from __future__ import annotations

import itertools
import json
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import dynecon
from dynecon.numerics import rouwenhorst, stationary_distribution, tauchen

OUT = Path(__file__).resolve().parents[1] / "crosscheck" / "results" / "quantecon.json"


def compare_discretization() -> dict[str, Any]:
    """Tauchen and Rouwenhorst: grids, transition matrices and stationary distributions."""
    from quantecon.markov import rouwenhorst as qe_rouwenhorst
    from quantecon.markov import tauchen as qe_tauchen

    methods = {
        "tauchen": (tauchen, qe_tauchen),
        "rouwenhorst": (rouwenhorst, qe_rouwenhorst),
    }
    worst: dict[str, float] = {"grid": 0.0, "P": 0.0, "stationary": 0.0}
    worst_case: dict[str, str] = {}
    n_cases = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quantecon warns about its rouwenhorst API change
        for (name, (ours, theirs)), rho, sigma, n in itertools.product(
            methods.items(), (0.0, 0.5, 0.9, 0.99), (0.05, 0.2), (2, 3, 5, 7, 15)
        ):
            grid, P = ours(rho, sigma, n)
            chain = theirs(n, rho, sigma)
            diffs = {
                "grid": float(np.abs(grid - chain.state_values).max()),
                "P": float(np.abs(P - chain.P).max()),
                "stationary": float(
                    np.abs(stationary_distribution(P) - chain.stationary_distributions[0]).max()
                ),
            }
            n_cases += 1
            for key, value in diffs.items():
                if value > worst[key]:
                    worst[key] = value
                    worst_case[key] = f"{name}, rho={rho}, sigma_e={sigma}, n={n}"
    tolerance = {"grid": 1e-12, "P": 1e-12, "stationary": 1e-8}
    return {
        "name": "Markov discretization (Tauchen, Rouwenhorst)",
        "compared": "dynecon.numerics.tauchen/rouwenhorst vs quantecon.markov.tauchen/rouwenhorst",
        "n_cases": n_cases,
        "max_abs_diff": worst,
        "worst_case": worst_case,
        "tolerance": tolerance,
        "passed": all(worst[k] <= tolerance[k] for k in worst),
        "notes": [
            "Argument order differs: Dynecon (rho, sigma_e, n_states), QuantEcon (n, rho, sigma).",
            "Both take sigma as the innovation standard deviation and use the same grid"
            " (+-3 unconditional sd for Tauchen, +-sd*sqrt(n-1) for Rouwenhorst).",
            "Stationary distributions differ most for near-unit-root chains (rho=0.99), where the"
            " problem is ill-conditioned; this is numerical conditioning, not a convention gap.",
        ],
    }


SECTIONS = (compare_discretization,)


def main() -> int:
    import quantecon

    results = [section() for section in SECTIONS]
    payload = {
        "dynecon": dynecon.__version__,
        "quantecon": quantecon.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "sections": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for r in results:
        print(f"{'ok' if r['passed'] else 'DIFFERS'}: {r['name']} ({r['n_cases']} cases)")
    print(f"wrote {OUT}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
