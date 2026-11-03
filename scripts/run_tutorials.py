"""Execute the code cells of ``docs/tutorials/*.md`` (MyST/jupytext) and fail on any error.

This is the lightweight check used by the test-suite; the full docs build executes the same
files through MyST-NB. Usage: ``python scripts/run_tutorials.py [NN ...]``.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "tutorials"
CELL = re.compile(r"```\{code-cell\}[^\n]*\n(.*?)```", re.S)


def cells(path: Path) -> list[str]:
    return CELL.findall(path.read_text(encoding="utf-8"))


def run(path: Path) -> float:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(TUTORIALS)
    namespace: dict[str, object] = {"__name__": "__tutorial__"}
    t0 = time.perf_counter()
    for i, source in enumerate(cells(path), 1):
        try:
            exec(compile(source, f"{path.name}[cell {i}]", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # report which cell failed
            raise RuntimeError(f"{path.name} cell {i} failed: {type(exc).__name__}: {exc}") from exc
    return time.perf_counter() - t0


def main() -> int:
    wanted = sys.argv[1:]
    files = sorted(
        p for p in TUTORIALS.glob("[0-9][0-9]_*.md") if not wanted or p.name[:2] in wanted
    )
    for path in files:
        seconds = run(path)
        print(f"{path.name}: ok ({seconds:.1f}s)")
        if seconds > 60:
            print(f"  warning: {path.name} exceeds the 60 s tutorial budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
