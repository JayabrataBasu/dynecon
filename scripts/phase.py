"""Print the operating protocol (Section 2) plus one phase of the execution plan."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1] / "docs" / "md" / "V1_EXECUTION_PLAN.md"


def extract(plan_text: str, phase: int) -> str:
    lines = plan_text.splitlines()
    out: list[str] = []
    start = next(i for i, ln in enumerate(lines) if ln.startswith("## 2. Operating protocol"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("## 3. v1.0 scope"))
    out.extend(lines[start:end])
    pat = re.compile(rf"^## P{phase} — ")
    ps = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if ps is None:
        raise SystemExit(f"phase P{phase} not found")
    pe = next(
        (
            i
            for i in range(ps + 1, len(lines))
            if lines[i].startswith("## P") or lines[i].startswith("# Part ")
        ),
        len(lines),
    )
    out.extend(lines[ps:pe])
    return "\n".join(out) + "\n"


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("usage: phase.py <n>", file=sys.stderr)
        return 2
    text = extract(PLAN.read_text(encoding="utf-8"), int(sys.argv[1]))
    sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
