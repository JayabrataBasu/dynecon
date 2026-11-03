"""Docstring audit: every public symbol has a docstring and an Economics note.

The note may sit in the symbol's own docstring or in its module docstring (a module-level note
covers the module's public symbols). A small allow-list covers pure plumbing.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Names that are constants/aliases or plumbing, not economic functions.
ALLOW = {
    "KNOWN_BACKENDS",
    "SOLVERS",
    "DEFAULTS",
    "PARAMS",
    "ANALYTIC",
    "CALIBRATION_SOURCE",
    "config",
    "__version__",
    "errors",
    "backends",
    "prange",
    "Solver",
    "Prior",
}


def public_modules() -> list[Any]:
    import econkit

    mods = [econkit]
    for info in pkgutil.walk_packages(econkit.__path__, "econkit."):
        parts = info.name.split(".")
        if (
            any(p.startswith("_") for p in parts)
            or "backends" in parts[:-1]
            or parts[-1] == "kernels"
        ):
            continue
        if parts[-1].startswith("jax") or parts[-1] in (
            "jax_kernels",
            "kalman_jax",
            "jacobian_jax",
        ):
            continue
        mods.append(importlib.import_module(info.name))
    return mods


def main() -> int:
    problems: list[str] = []
    for mod in public_modules():
        names = getattr(mod, "__all__", [])
        mod_doc = inspect.getdoc(mod) or ""
        for name in names:
            if name in ALLOW:
                continue
            obj = getattr(mod, name, None)
            if obj is None or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", "").split(".")[0] != "econkit":
                continue
            doc = inspect.getdoc(obj) or ""
            defining = importlib.import_module(obj.__module__)
            if not doc:
                problems.append(f"{mod.__name__}.{name}: no docstring")
            elif (
                "Economics note" not in doc
                and "Economics note" not in (inspect.getdoc(defining) or "")
                and "Economics note" not in mod_doc
            ):
                problems.append(
                    f"{mod.__name__}.{name}: no 'Economics note' in docstring or module docstring"
                )
    if problems:
        print("docstring audit failed:")
        print("\n".join(sorted(set(problems))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
