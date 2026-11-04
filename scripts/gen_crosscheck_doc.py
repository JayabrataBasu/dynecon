"""Render ``docs/md/crosscheck.md`` from ``crosscheck/results/quantecon.json`` (no recomputation)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "crosscheck" / "results" / "quantecon.json"
DST = ROOT / "docs" / "md" / "crosscheck.md"


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
    DST.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
