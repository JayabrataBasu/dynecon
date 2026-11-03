"""Fail if the plan's step headings and V1_PROGRESS.md checklist drift apart."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "md" / "V1_EXECUTION_PLAN.md"
PROGRESS = ROOT / "docs" / "md" / "V1_PROGRESS.md"


def plan_steps(text: str) -> set[str]:
    steps = set()
    for ln in text.splitlines():
        m = re.match(r"^### (P\d+\.\d+)\b(.*)$", ln)
        if m and "retired" not in m.group(2):
            steps.add(m.group(1))
    return steps


def progress_steps(text: str) -> set[str]:
    return set(re.findall(r"^- \[[ xX]\] (P\d+\.\d+)\b", text, flags=re.M))


def main() -> int:
    plan = plan_steps(PLAN.read_text(encoding="utf-8"))
    prog = progress_steps(PROGRESS.read_text(encoding="utf-8"))
    missing = sorted(plan - prog)
    extra = sorted(prog - plan)
    if missing or extra:
        print("progress check failed")
        if missing:
            print("  in plan but not in V1_PROGRESS.md:", ", ".join(missing))
        if extra:
            print("  in V1_PROGRESS.md but not in plan:", ", ".join(extra))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
