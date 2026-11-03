from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase_extract_contains_protocol_and_phase_only() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "phase.py"), "5"],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout
    assert "## P5 — " in out
    assert "R21" in out
    assert "## P6 — " not in out
