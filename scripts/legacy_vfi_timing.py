"""Time the pre-rewrite pure-Python VFI (git tag ``v0.1.0-baseline``) per Bellman iteration.

The tagged baseline is checked out into a temporary worktree and run in a subprocess, so no
legacy code lives in the repository (plan R21a, P5.10). The legacy solver is far too slow to
run to convergence at (200, 7), so the cost of a fixed number of iterations is measured.

Usage: ``python scripts/legacy_vfi_timing.py [n_assets] [n_iterations]`` prints JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from econkit.model import AR1Shock, Control, Model, Param, State
from econkit.solvers import VFIConfig, solve_consumption_savings_stochastic_vfi
n_a, iters = int(sys.argv[2]), int(sys.argv[3])
m = Model("legacy")
m.params(beta=Param(0.96), sigma=Param(2.0), r=Param(0.03), y=Param(1.0))
m.add_state(State("assets", bounds=(0.0, 50.0), n_points=n_a))
m.add_state(State("income", shock=AR1Shock(rho=0.9, sigma_e=0.1, n_states=7)))
m.add_control(Control("consumption"))
@m.equation("bellman")
def b(v, p):
    return v
t = time.perf_counter()
sol = solve_consumption_savings_stochastic_vfi(m.compile(), config=VFIConfig(tol=1e-30, max_iter=iters))
dt = time.perf_counter() - t
print(json.dumps({"seconds": dt, "iterations": sol.iterations, "per_iteration": dt / sol.iterations}))
"""


def main() -> int:
    n_a = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "baseline"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), "v0.1.0-baseline"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        try:
            out = subprocess.run(
                [sys.executable, "-c", CODE, str(wt), str(n_a), str(iters)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt)], cwd=root, capture_output=True
            )
    result = json.loads(out.strip().splitlines()[-1])
    result.update(n_assets=n_a, n_shocks=7)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
