"""Provenance record: versions, platform, git commit, timestamp.

Economics note:
Recording the software and machine behind a result is part of making it reproducible.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from econkit._version import __version__

__all__ = ["collect_provenance"]


def _version(module: str) -> str | None:
    try:
        return str(importlib.import_module(module).__version__)
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def collect_provenance() -> dict[str, Any]:
    """Return versions of the stack, platform, git commit (if any) and a UTC timestamp."""
    return {
        "econkit": __version__,
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "numba": _version("numba"),
        "jax": _version("jax"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
