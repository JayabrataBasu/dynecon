"""Enforce rule R21e: NEAR-FUTURE tags must name a live step within the review window."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "md" / "V1_EXECUTION_PLAN.md"
PROGRESS = ROOT / "docs" / "md" / "V1_PROGRESS.md"
TAG = re.compile(r"NEAR-FUTURE\(P(\d+)\.(\d+)\)")
WINDOWS = [(0, 6), (7, 11), (12, 16), (17, 17)]


def window(phase: int) -> int:
    for i, (lo, hi) in enumerate(WINDOWS):
        if lo <= phase <= hi:
            return i
    raise ValueError(phase)


def main() -> int:
    plan = PLAN.read_text(encoding="utf-8")
    prog = PROGRESS.read_text(encoding="utf-8")
    live: set[str] = set()
    for ln in plan.splitlines():
        m = re.match(r"^### (P\d+\.\d+)\b(.*)$", ln)
        if m and "retired" not in m.group(2):
            live.add(m.group(1))
    ticked = set(re.findall(r"^- \[[xX]\] (P\d+\.\d+)\b", prog, flags=re.M))
    unticked = sorted(
        (s for s in live if s not in ticked),
        key=lambda s: tuple(int(x) for x in s[1:].split(".")),
    )
    current = window(int(unticked[0][1:].split(".")[0])) if unticked else len(WINDOWS) - 1
    bad = []
    for path in (ROOT / "dynecon").rglob("*.py"):
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = TAG.search(line)
            if not m:
                continue
            step = f"P{m.group(1)}.{m.group(2)}"
            where = f"{path.relative_to(ROOT)}:{no}"
            if step not in live:
                bad.append(f"{where}: {step} does not exist or is retired")
            elif step in ticked:
                bad.append(f"{where}: {step} already ticked; code should have a caller")
            elif window(int(m.group(1))) > current + 1:
                bad.append(f"{where}: {step} is more than one review window ahead")
    if bad:
        print("\n".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
