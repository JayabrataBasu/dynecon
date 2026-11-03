"""Portable equivalent of ``make ci`` (works without ``make``, e.g. on Windows).

Runs: layout + progress checks, ruff, mypy, pytest, vulture, usage / near-future / docstring checks.
Usage: ``python scripts/ci.py [--coverage] [--skip-tests]``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(name: str, *cmd: str) -> bool:
    print(f"== {name}", flush=True)
    argv = [PY, *cmd] if cmd[0].endswith(".py") else [PY, "-m", *cmd]
    ok = subprocess.run(argv, cwd=ROOT).returncode == 0
    print(f"-- {name}: {'ok' if ok else 'FAILED'}", flush=True)
    return ok


def main() -> int:
    coverage = "--coverage" in sys.argv
    skip_tests = "--skip-tests" in sys.argv
    pytest = ["pytest", "-p", "no:warnings"] + (
        ["--cov", "--cov-report=term-missing"] if coverage else []
    )
    steps = [
        ("test layout", "scripts/check_test_layout.py"),
        ("progress", "scripts/check_progress.py"),
        ("ruff", "ruff", "check", "."),
        ("mypy", "mypy", "econkit"),
        ("pytest", *pytest),
        ("vulture", "vulture", "econkit", "scripts/vulture_whitelist.py"),
        ("usage", "scripts/check_usage.py"),
        ("near-future", "scripts/check_near_future.py"),
        ("docstrings", "scripts/check_docstrings.py"),
    ]
    failed = [
        s[0] for s in steps if not (skip_tests and s[0] == "pytest") and not run(s[0], *s[1:])
    ]
    if failed:
        print("FAILED steps:", ", ".join(failed))
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
