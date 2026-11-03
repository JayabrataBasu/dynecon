"""Write (or check) the public API surface: every name in ``__all__`` of the public modules.

Usage: ``python scripts/dump_api.py`` rewrites ``docs/md/API_SURFACE.txt``;
``python scripts/dump_api.py --check`` exits non-zero if the surface changed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "md" / "API_SURFACE.txt"

MODULES = [
    "dynecon",
    "dynecon.model",
    "dynecon.models",
    "dynecon.numerics",
    "dynecon.simulation",
    "dynecon.estimation",
    "dynecon.experiment",
    "dynecon.data",
    "dynecon.viz",
    "dynecon.solvers",
    "dynecon.solvers.perturbation",
    "dynecon.solvers.equilibrium",
    "dynecon.solvers.batch",
    "dynecon.core.errors",
]


def surface() -> str:
    lines = []
    for name in MODULES:
        mod = importlib.import_module(name)
        for sym in sorted(getattr(mod, "__all__", [])):
            lines.append(f"{name}.{sym}")
    return "\n".join(lines) + "\n"


def main() -> int:
    current = surface()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != current:
            print("public API surface changed; run scripts/dump_api.py and update the changelog")
            return 1
        return 0
    OUT.write_text(current, encoding="utf-8")
    print(f"wrote {OUT} ({len(current.splitlines())} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
