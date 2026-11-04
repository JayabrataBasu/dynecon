"""Benchmark Dynecon against other libraries on identical problems.

Run inside an environment that has dynecon, quantecon and sequence-jacobian installed (for example a
throw-away venv); writes ``crosscheck/results/libs_benchmark.json``. Rules, so the numbers are fair:

* the same problem: Dynecon's asset grid, income values and transition matrix are passed to the other
  library, and accuracy against Dynecon is reported next to every timing;
* timings are wall-clock ``perf_counter`` seconds, minimum and median of ``REPEATS`` warm runs; the first
  call (which includes JIT compilation) is reported separately;
* tolerances are matched where the libraries allow it, and each library's own defaults are also timed.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import warnings
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import dynecon as ek
from dynecon import models
from dynecon.simulation import stationary_distribution_young

OUT = Path(__file__).resolve().parents[1] / "crosscheck" / "results" / "libs_benchmark.json"
REPEATS = 7
warnings.simplefilter("ignore")


def timed(fn: Callable[[], Any]) -> dict[str, float]:
    """First-call time, then min/median over warm repeats."""
    t0 = perf_counter()
    fn()
    first = perf_counter() - t0
    runs = []
    for _ in range(REPEATS):
        t0 = perf_counter()
        fn()
        runs.append(perf_counter() - t0)
    return {"first_call_s": first, "min_s": min(runs), "median_s": statistics.median(runs)}


def household_vs_ssj(n_a: int, n_z: int = 7) -> dict[str, Any]:
    """Household EGM + stationary distribution: Dynecon vs Sequence-Space Jacobian ``hh`` block."""
    from sequence_jacobian.hetblocks.hh_sim import hh

    model = models.load(
        "consumption_savings", n_a=n_a, n_z=n_z, sigma=2.0, r=0.03, beta=0.96
    ).compile()
    income = model.shock_process("income")
    grid, y, P = model.grids["assets"], income.values, income.P
    calib = {"a_grid": grid, "y": y, "Pi": P, "r": 0.03, "beta": 0.96, "eis": 0.5}

    def ssj() -> Any:
        return hh.steady_state(calib)

    def dyn(backend: str, egm_tol: float, dist_tol: float) -> Callable[[], Any]:
        def run() -> Any:
            sol = ek.solve(model, method="egm", backend=backend, tol=egm_tol)
            return sol, stationary_distribution_young(sol, tol=dist_tol, backend=backend)

        return run

    ref = ssj()
    sol, mu = dyn("numba", 1e-8, 1e-10)()
    accuracy = {
        "policy_max_abs_diff": float(
            np.abs(ref.internals["hh"]["a"] - sol.policy["assets_next"].T).max()
        ),
        "distribution_max_abs_diff": float(np.abs(ref.internals["hh"]["D"] - mu.T).max()),
        "aggregate_assets_ssj": float(ref["A"]),
        "aggregate_assets_dynecon": float((mu * sol.policy["assets_next"]).sum()),
    }
    return {
        "problem": f"consumption-savings, CRRA 2, r=0.03, beta=0.96, AR(1) income, n_a={n_a}, n_z={n_z}",
        "n_a": n_a,
        "n_z": n_z,
        "accuracy_vs_ssj_at_matched_tolerance": accuracy,
        "timings": {
            "ssj_defaults (backward 1e-8, forward 1e-10)": timed(ssj),
            "dynecon_numba matched (1e-8, 1e-10)": timed(dyn("numba", 1e-8, 1e-10)),
            "dynecon_numpy matched (1e-8, 1e-10)": timed(dyn("numpy", 1e-8, 1e-10)),
            "dynecon_numba defaults (1e-10, 1e-12)": timed(dyn("numba", 1e-10, 1e-12)),
        },
    }


def savings_vs_quantecon(n_a: int, n_z: int = 3, beta: float = 0.96) -> dict[str, Any]:
    """Discrete savings problem: Dynecon VFI vs QuantEcon ``DiscreteDP`` (same grid, same answer)."""
    from quantecon.markov import DiscreteDP

    model = models.load(
        "consumption_savings", n_a=n_a, n_z=n_z, sigma=2.0, r=0.03, beta=beta
    ).compile()
    income = model.shock_process("income")
    a, z, P = model.grids["assets"], income.values, income.P
    n_s = n_a * n_z

    def build() -> DiscreteDP:
        reward = np.full((n_s, n_a), -np.inf)
        trans = np.zeros((n_s, n_a, n_s))
        for i in range(n_a):
            for j in range(n_z):
                cons = 1.03 * a[i] + z[j] - a
                ok = cons > 0
                reward[i * n_z + j, ok] = cons[ok] ** (-1.0) / (-1.0)
                for k in range(n_a):
                    trans[i * n_z + j, k, k * n_z : (k + 1) * n_z] = P[j]
        return DiscreteDP(reward, trans, beta)

    ddp = build()

    def qe_policy_iteration() -> Any:
        return ddp.solve(method="policy_iteration", epsilon=1e-8, max_iter=1000)

    def qe_value_iteration() -> Any:
        return ddp.solve(method="value_iteration", epsilon=1e-8, max_iter=5000)

    def dyn(backend: str) -> Callable[[], Any]:
        return lambda: ek.solve(model, method="vfi", backend=backend, tol=1e-8)

    sol = dyn("numba")()
    res = qe_policy_iteration()
    accuracy = {
        "value_max_abs_diff": float(np.abs(res.v.reshape(n_a, n_z) - sol.value).max()),
        "policy_mismatches": int(
            (res.sigma.reshape(n_a, n_z) != sol.policy["assets_next_index"]).sum()
        ),
        "n_states": n_s,
    }
    return {
        "problem": f"discrete savings (CRRA 2, r=0.03, beta={beta}), n_a={n_a}, n_z={n_z}",
        "n_a": n_a,
        "n_z": n_z,
        "accuracy_vs_quantecon": accuracy,
        "timings": {
            "quantecon_policy_iteration (solve only, 1e-8)": timed(qe_policy_iteration),
            "quantecon_value_iteration (solve only, 1e-8)": timed(qe_value_iteration),
            "dynecon_vfi_numba (1e-8)": timed(dyn("numba")),
            "dynecon_vfi_numpy (1e-8)": timed(dyn("numpy")),
        },
        "quantecon_model_build_s": timed(build)["min_s"],
    }


def main() -> int:
    import quantecon
    import sequence_jacobian

    cpu = platform.processor()
    payload = {
        "versions": {
            "dynecon": ek.__version__,
            "quantecon": quantecon.__version__,
            "sequence_jacobian": getattr(sequence_jacobian, "__version__", "1.0.0"),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "machine": {"platform": platform.platform(), "cpu": cpu, "logical_cores": os.cpu_count()},
        "repeats": REPEATS,
        "household_vs_ssj": [household_vs_ssj(n) for n in (200, 500, 1000)],
        "savings_vs_quantecon": [savings_vs_quantecon(n) for n in (60, 100, 150)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
