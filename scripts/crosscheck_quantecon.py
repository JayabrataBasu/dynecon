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
from dynecon.estimation import kalman_filter
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


def compare_kalman() -> dict[str, Any]:
    """Kalman filter: predicted/filtered states, covariances and the Gaussian log-likelihood.

    QuantEcon's ``Kalman`` gives the recursions but no likelihood, so the likelihood is scored from
    QuantEcon's innovations with ``scipy.stats.multivariate_normal`` (independent of Dynecon's formula).
    """
    from quantecon import Kalman, LinearStateSpace
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(0)
    worst = {"predicted_state": 0.0, "predicted_cov": 0.0, "filtered_state": 0.0, "loglik": 0.0}
    n_cases = 0
    for n_a, n_y in ((1, 1), (2, 1), (3, 2), (4, 3)):
        for _ in range(5):
            A = rng.normal(size=(n_a, n_a))
            A *= 0.9 / np.abs(np.linalg.eigvals(A)).max()
            C = 0.3 * rng.normal(size=(n_a, n_a))
            G = rng.normal(size=(n_y, n_a))
            Hf = 0.4 * np.eye(n_y)
            Q, R = C @ C.T, Hf @ Hf.T  # Dynecon takes covariances; QuantEcon takes factors
            a0, P0 = np.zeros(n_a), np.eye(n_a)
            x, y = rng.multivariate_normal(a0, P0), []
            for _t in range(80):
                y.append(G @ x + Hf @ rng.normal(size=n_y))
                x = A @ x + C @ rng.normal(size=n_a)
            ours = kalman_filter(np.array(y), G, R, A, Q, a0, P0)
            kal = Kalman(LinearStateSpace(A, C, G, H=Hf), x_hat=a0.copy(), Sigma=P0.copy())
            pred_s, pred_c, filt_s, loglik = [], [], [], 0.0
            for yt in y:
                prior = np.asarray(kal.x_hat).reshape(n_a)
                pred_s.append(prior.copy())
                pred_c.append(np.array(kal.Sigma))
                F = G @ kal.Sigma @ G.T + R
                loglik += multivariate_normal(mean=np.zeros(n_y), cov=F).logpdf(yt - G @ prior)
                kal.prior_to_filtered(yt)
                filt_s.append(np.asarray(kal.x_hat).reshape(n_a).copy())
                kal.filtered_to_forecast()
            diffs = {
                "predicted_state": np.abs(np.array(pred_s) - ours.predicted_state).max(),
                "predicted_cov": np.abs(np.array(pred_c) - ours.predicted_cov).max(),
                "filtered_state": np.abs(np.array(filt_s) - ours.filtered_state).max(),
                "loglik": abs(loglik - ours.loglik),
            }
            n_cases += 1
            for key, value in diffs.items():
                worst[key] = max(worst[key], float(value))
    tolerance = {k: 1e-10 for k in worst}
    return {
        "name": "Kalman filter",
        "compared": "dynecon.estimation.kalman_filter vs quantecon.Kalman (+ scipy likelihood)",
        "n_cases": n_cases,
        "max_abs_diff": worst,
        "worst_case": {},
        "tolerance": tolerance,
        "passed": all(worst[k] <= tolerance[k] for k in worst),
        "notes": [
            "Random stable systems (1-4 states, 1-3 observables, 80 periods, fixed seed).",
            "Dynecon takes covariances (H, Q); QuantEcon takes factors (C, H) with covariances"
            " CC' and HH'. Both treat the initial (a0, P0) as the prior for the first period.",
            "The log-likelihood includes the full Gaussian constant.",
        ],
    }


SECTIONS = (compare_discretization, compare_kalman)


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
