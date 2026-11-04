"""Render ``docs/md/crosscheck.md`` from ``crosscheck/results/quantecon.json`` (no recomputation)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "crosscheck" / "results" / "quantecon.json"
DST = ROOT / "docs" / "md" / "crosscheck.md"
BENCH = ROOT / "crosscheck" / "results" / "libs_benchmark.json"


def _best(timings: dict[str, dict[str, float]], prefix: str) -> float:
    return next(t["min_s"] for k, t in timings.items() if k.startswith(prefix))


def benchmark_lines() -> list[str]:
    """Performance section rendered from ``libs_benchmark.json`` (ratios computed, never typed)."""
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    v, m = data["versions"], data["machine"]
    lines = [
        "# Performance and accuracy against other libraries",
        "",
        "Generated from `crosscheck/results/libs_benchmark.json` by `scripts/gen_crosscheck_doc.py`;",
        "regenerate the data with `scripts/benchmark_vs_libs.py`. Wall-clock seconds, minimum of "
        f"{data['repeats']} warm runs; first-call (JIT) times are in the JSON. Single machine "
        f"({m['platform']}, {m['logical_cores']} logical cores), Python {v['python']}, "
        f"dynecon {v['dynecon']}, quantecon {v['quantecon']}, sequence-jacobian {v['sequence_jacobian']}, "
        f"numpy {v['numpy']}. Both libraries receive the same grid, income values and transition "
        "matrix; one machine and one problem family, so treat the ratios as indicative.",
        "",
        "## Household EGM + stationary distribution vs Sequence-Space Jacobian (SSJ)",
        "",
        "SSJ's `hh` block is also an endogenous-grid solver, so this compares like with like. "
        "Tolerances are matched (policy 1e-8, distribution 1e-10).",
        "",
        "| assets | SSJ (s) | Dynecon Numba (s) | Dynecon NumPy (s) | Dynecon / SSJ (Numba) | policy diff | distribution diff |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in data["household_vs_ssj"]:
        t, acc = r["timings"], r["accuracy_vs_ssj_at_matched_tolerance"]
        ssj, numba, numpy_ = (
            _best(t, "ssj"),
            _best(t, "dynecon_numba matched"),
            _best(t, "dynecon_numpy"),
        )
        lines.append(
            f"| {r['n_a']} | {ssj:.3f} | {numba:.3f} | {numpy_:.3f} | {numba / ssj:.1f}x | "
            f"{acc['policy_max_abs_diff']:.1e} | {acc['distribution_max_abs_diff']:.1e} |"
        )
    lines += [
        "",
        "SSJ is faster than Dynecon on this task. The two agree to about 2e-7 in the policy and 3e-9 in the "
        "distribution, which is the size expected from iteration tolerances of 1e-8 and 1e-10.",
        "",
        "## Discrete savings problem vs QuantEcon `DiscreteDP`",
        "",
        "Same discrete-choice problem and grid (see the cross-check above), tolerance 1e-8, "
        "model construction excluded for QuantEcon. `DiscreteDP` is a generic solver that does "
        "not exploit the monotone structure of the savings policy.",
        "",
        "| states | QuantEcon policy iter. (s) | QuantEcon value iter. (s) | Dynecon VFI Numba (s) | speed-up vs policy iter. | policy mismatches |",
        "|---|---|---|---|---|---|",
    ]
    for r in data["savings_vs_quantecon"]:
        t, acc = r["timings"], r["accuracy_vs_quantecon"]
        pi, vi, dy = (
            _best(t, "quantecon_policy"),
            _best(t, "quantecon_value"),
            _best(t, "dynecon_vfi_numba"),
        )
        lines.append(
            f"| {acc['n_states']} | {pi:.3f} | {vi:.3f} | {dy:.3f} | {pi / dy:.1f}x | {acc['policy_mismatches']} |"
        )
    lines += [
        "",
        "Not benchmarked: HARK (its income process and normalisation differ, so no identical problem "
        "has been set up yet) and econpizza/Dynare (different problem classes).",
        "",
    ]
    return lines


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    lines = [
        "# Cross-check against QuantEcon",
        "",
        "Generated from `crosscheck/results/quantecon.json` by `scripts/gen_crosscheck_doc.py`;",
        "regenerate the data with `python scripts/crosscheck_quantecon.py`. Only numbers in this",
        "file may be quoted elsewhere.",
        "",
        f"- Versions: dynecon {data['dynecon']}, quantecon {data['quantecon']}, "
        f"numpy {data['numpy']}, python {data['python']}",
        "",
        "[QuantEcon](https://quantecon.org) is an established open-source library for"
        " computational economics. Agreement with an independent implementation on identical"
        " inputs is evidence that Dynecon's results are right; a disagreement is either a"
        " documented convention difference or a bug to fix.",
        "",
    ]
    for sec in data["sections"]:
        verdict = "agrees within tolerance" if sec["passed"] else "**differs**"
        lines += [f"## {sec['name']}: {verdict}", "", f"Compared: {sec['compared']}.", ""]
        lines += [f"{sec['n_cases']} parameter combinations.", ""]
        lines += [
            "| quantity | largest abs. difference | tolerance | worst case |",
            "|---|---|---|---|",
        ]
        for key, value in sec["max_abs_diff"].items():
            lines.append(
                f"| {key} | {value:.2e} | {sec['tolerance'][key]:.0e} | {sec['worst_case'].get(key, '-')} |"
            )
        lines += [""] + [f"- {note}" for note in sec["notes"]] + [""]
    if BENCH.exists():
        lines += benchmark_lines()
    DST.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
