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
from dynecon import models
from dynecon.estimation import kalman_filter
from dynecon.numerics import rouwenhorst, stationary_distribution, tauchen
from dynecon.simulation import stationary_distribution_young

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


def _household_case(n_a: int, n_z: int, sigma: float, r: float, beta: float, a_min: float) -> dict:
    """One savings problem solved by Dynecon VFI and by QuantEcon ``DiscreteDP`` on the same grid."""
    from quantecon.markov import DiscreteDP

    model = models.load(
        "consumption_savings", n_a=n_a, n_z=n_z, sigma=sigma, r=r, beta=beta, a_min=a_min
    ).compile()
    sol = dynecon.solve(model, method="vfi", tol=1e-12, max_iter=20000)
    a = model.grids["assets"]
    income = model.shock_process("income")
    z, P = income.values, income.P
    n_s = n_a * n_z
    reward = np.full((n_s, n_a), -np.inf)  # state s = i * n_z + j, action k = next-asset index
    trans = np.zeros((n_s, n_a, n_s))
    for i in range(n_a):
        for j in range(n_z):
            cons = (1 + r) * a[i] + z[j] - a  # independent CRRA / budget, not Dynecon's code
            ok = cons > 0
            reward[i * n_z + j, ok] = (
                np.log(cons[ok]) if sigma == 1 else cons[ok] ** (1 - sigma) / (1 - sigma)
            )
            for k in range(n_a):
                trans[i * n_z + j, k, k * n_z : (k + 1) * n_z] = P[j]
    ddp = DiscreteDP(reward, trans, beta)
    res = ddp.solve(method="policy_iteration", epsilon=1e-12, max_iter=2000)
    chain = ddp.controlled_mc(res.sigma)
    mu = stationary_distribution_young(sol)
    return {
        "value": float(np.abs(res.v.reshape(n_a, n_z) - sol.value).max()),
        "policy_mismatches": int(
            (res.sigma.reshape(n_a, n_z) != sol.policy["assets_next_index"]).sum()
        ),
        "stationary_residual": float(np.abs(mu.ravel() @ chain.P - mu.ravel()).max()),
        "mass_error": float(abs(mu.sum() - 1.0)),
        "recurrent_classes": len(chain.recurrent_classes),
        "n_states": n_s,
    }


def compare_household() -> dict[str, Any]:
    """Savings problem: Dynecon VFI vs QuantEcon ``DiscreteDP`` (value, policy, stationary dist.)."""
    cases = [
        (60, 3, 2.0, 0.03, 0.96, 0.0),
        (80, 5, 1.5, 0.02, 0.95, 0.0),
        (60, 3, 3.0, 0.03, 0.96, -1.0),
    ]
    results = [_household_case(*c) for c in cases]
    worst = {
        "value": max(r["value"] for r in results),
        "policy_mismatches": float(sum(r["policy_mismatches"] for r in results)),
        "stationary_residual": max(r["stationary_residual"] for r in results),
        "mass_error": max(r["mass_error"] for r in results),
    }
    tolerance = {
        "value": 1e-9,
        "policy_mismatches": 0.0,
        "stationary_residual": 1e-10,
        "mass_error": 1e-12,
    }
    total = sum(r["n_states"] for r in results)
    return {
        "name": "Household savings problem (VFI)",
        "compared": "dynecon.solve(method='vfi') + stationary_distribution_young vs"
        " quantecon.markov.DiscreteDP (policy iteration) + controlled_mc stationary distribution",
        "n_cases": len(cases),
        "max_abs_diff": worst,
        "worst_case": {
            "policy_mismatches": f"{int(worst['policy_mismatches'])} of {total} states",
            "stationary_residual": "recurrent classes per case: "
            + ", ".join(str(r["recurrent_classes"]) for r in results),
        },
        "tolerance": tolerance,
        "passed": all(worst[k] <= tolerance[k] for k in worst),
        "notes": [
            "QuantEcon 0.13 has no household/Aiyagari class, so the same discrete-choice problem"
            " (state = assets x income, action = next-asset grid index) is built for `DiscreteDP`"
            " with an independently written CRRA utility and budget constraint.",
            "Inputs shared by construction: Dynecon's asset grid, income values and transition"
            " matrix. Cases: CRRA 2.0 / 1.5 / 3.0, r 0.02-0.03, beta 0.95-0.96, borrowing limit"
            " 0 and -1.",
            "Policy mismatches are counted over grid indices (0 means identical choices).",
            "On a discrete grid the optimal-policy chain has several recurrent classes, so its"
            " stationary distribution is not unique and cannot be compared point by point. The"
            " check is instead that Dynecon's distribution is invariant under QuantEcon's own"
            " controlled chain (mu P = mu) and sums to one.",
            "The EGM solver is continuous-choice, so it is not expected to match a discrete-grid"
            " solution exactly; its accuracy is covered by the Euler-error diagnostics.",
        ],
    }


SECTIONS = (compare_discretization, compare_kalman, compare_household)


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
