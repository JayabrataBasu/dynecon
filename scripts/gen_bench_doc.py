"""Turn the newest ``benchmarks/results/*.json`` into ``docs/md/benchmarks.md``.

Only numbers in that file may be quoted in the README or documentation (plan R19).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ("bellman_cs", "numba", "numpy", 20.0, "Numba >= 20x NumPy vectorised"),
    ("simulate_panel", "numba", "numpy", 10.0, "Numba >= 10x NumPy loop"),
    ("young_stationary", "numba", "numpy", 3.0, "Numba >= 3x NumPy histogram step"),
]


def _legacy() -> dict[str, float] | None:
    try:
        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "legacy_vfi_timing.py"), "200", "2"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout
        return json.loads(out.strip().splitlines()[-1])
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
        return None


def _new_iteration_seconds(n_a: int = 200, n_z: int = 7, repeats: int = 30) -> float:
    """Minimum time of one Numba Bellman maximisation at the legacy benchmark's grid size."""
    import time

    import numpy as np

    from dynecon import models
    from dynecon.numerics._dispatch import resolve_backend
    from dynecon.solvers.dp.vfi import build_payoff_tensor

    model = models.load("consumption_savings", n_a=n_a, n_z=n_z, a_max=50.0).compile()
    U = build_payoff_tensor(model)
    evt = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal((n_z, n_a)).cumsum(axis=1) * 0.01
    )
    _, kernel = resolve_backend("bellman_saving", "numba")
    kernel(U, evt, True)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        kernel(U, evt, True)
        times.append(time.perf_counter() - t)
    return min(times)


def main() -> int:
    files = sorted((ROOT / "benchmarks" / "results").glob("*.json"))
    if not files:
        print("no benchmark results found", file=sys.stderr)
        return 1
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    recs = data["records"]
    by = {(r["name"], r["backend"]): r for r in recs}
    m = data["machine"]
    lines = [
        "# Benchmarks",
        "",
        f"Generated from `{files[-1].name}` by `scripts/gen_bench_doc.py`. "
        "Timings are the **minimum** of the repeats; compile time is the first call minus the "
        "steady-state median. All numbers come from one machine; never compare across machines.",
        "",
        f"- Machine: {m['processor']}, {m['cores']} logical cores, RAM {m.get('ram_gb')} GB, {m['platform']}",
        "- Versions: " + ", ".join(f"{k} {v}" for k, v in m["versions"].items()),
        f"- Date: {data['date']}",
        "",
        "| case | backend | min (ms) | speedup vs NumPy | compile (s) | max abs diff vs NumPy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in recs:
        ref = by.get((r["name"], "numpy"))
        speed = "" if ref is None or r["backend"] == "numpy" else f"{ref['min'] / r['min']:.2f}x"
        diff = "" if r["max_abs_diff_vs_numpy"] is None else f"{r['max_abs_diff_vs_numpy']:.1e}"
        lines.append(
            f"| {r['name']} | {r['backend']} | {r['min'] * 1e3:.2f} | {speed} | "
            f"{r['compile_seconds']:.2f} | {diff} |"
        )
    lines += ["", "## Goals versus measurements", "", "Goals are never edited to fit results.", ""]
    for name, fast, slow, goal, text in TARGETS:
        a, b = by.get((name, fast)), by.get((name, slow))
        if a and b:
            ratio = b["min"] / a["min"]
            verdict = "met" if ratio >= goal else "**missed**"
            lines.append(f"- `{name}` ({text}): measured {ratio:.2f}x, goal {goal:g}x: {verdict}")
    vfi_np, vfi_nb = by.get(("vfi_cs", "numpy")), by.get(("vfi_cs", "numba"))
    if vfi_np and vfi_nb:
        ratio = vfi_np["min"] / vfi_nb["min"]
        lines.append(
            f"- `vfi_cs` (full solve, Numba monotone + Howard vs vectorised NumPy): {ratio:.2f}x, "
            f"goal 5x: {'met' if ratio >= 5 else '**missed**'}"
        )
    egm_nb = by.get(("egm_cs", "numba"))
    if egm_nb and vfi_nb:
        ratio = vfi_nb["min"] / egm_nb["min"]
        lines.append(
            f"- `egm_cs` (EGM at n_a=1000 vs Numba VFI at n_a=500): {ratio:.2f}x faster, goal 20x: "
            f"{'met' if ratio >= 20 else '**missed**'}"
        )
        egm_same = by.get(("egm_cs_500", "numba"))
        if egm_same and vfi_nb:
            same = vfi_nb["min"] / egm_same["min"]
            lines.append(
                f"- `egm_cs_500` (EGM vs Numba VFI on the same grid, n_a=500): EGM is {same:.2f}x as fast "
                f"as VFI (above 1 means EGM is faster)"
            )
        if ratio < 20:
            lines += [
                "",
                "**Status of the EGM goal: missed, and retired (maintainer decision, 2026-11-05).** The "
                "20x goal was set before anything was measured and compares EGM on a grid twice as large "
                "with VFI at n_a=500. It was missed in every recorded run and is not reachable by tuning: "
                "after Anderson acceleration (about half the iterations) EGM still spends its time in "
                "Python-level array passes, while the Numba VFI with Howard acceleration and monotone "
                "search needs few outer iterations. The line is kept here so the miss stays visible; no "
                "replacement goal has been set. A fused all-Numba EGM loop is untried and may help, "
                "but is not expected to reach 20x. EGM's advantages in this library are a continuous "
                "choice (accuracy) and no root-finding, not speed on small grids.",
            ]
    legacy = _legacy()
    if legacy:
        new_it = _new_iteration_seconds(legacy["n_assets"], legacy["n_shocks"])
        ratio = legacy["per_iteration"] / new_it
        lines += [
            "",
            "### Legacy pure-Python VFI (git tag `v0.1.0-baseline`)",
            "",
            f"One Bellman iteration at (n_a, n_z) = ({legacy['n_assets']}, {legacy['n_shocks']}): legacy "
            f"{legacy['per_iteration']:.3f} s (measured over {legacy['iterations']} iterations in a temporary "
            f"worktree), Numba kernel {new_it * 1e3:.3f} ms: **{ratio:.0f}x** per iteration (goal 50x for the full "
            f"solve; Howard acceleration and monotone search reduce the iteration count further).",
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- `simulate_panel` uses 2x10^4 agents (not 10^5) to keep peak memory modest.",
        "- JAX is not claimed to be faster on CPU; it exists for autodiff, `vmap` and GPU.",
        "- Timing is never a required CI check.",
    ]
    out = ROOT / "docs" / "md" / "benchmarks.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
