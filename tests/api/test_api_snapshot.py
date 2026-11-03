"""The public API surface is frozen in docs/md/API_SURFACE.txt (regenerate deliberately)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_api_surface_matches_snapshot() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "dump_api.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout
