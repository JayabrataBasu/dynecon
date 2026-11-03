"""Backend availability and introspection.

Economics note:
Knowing which numerical engine ran (and which libraries produced the answer) is
part of making a computed result reproducible.
"""

from __future__ import annotations

import importlib.util

from econkit.utils.jit import NUMBA_AVAILABLE

__all__ = ["KNOWN_BACKENDS", "available_backends", "describe", "is_available"]

KNOWN_BACKENDS: tuple[str, ...] = ("numpy", "numba", "jax")


def is_available(name: str) -> bool:
    """Whether the backend's library can be imported."""
    if name == "numpy":
        return True
    if name == "numba":
        return NUMBA_AVAILABLE
    if name == "jax":
        return importlib.util.find_spec("jax") is not None
    return False


def available_backends() -> list[str]:
    """Known backends that are installed, NumPy first."""
    return [b for b in KNOWN_BACKENDS if is_available(b)]


def describe() -> str:
    """Operation x backend table plus library versions (paste into bug reports)."""
    import econkit.numerics  # noqa: F401  (ensures registration)
    from econkit.numerics._registry import list_registered
    from econkit.utils.provenance import collect_provenance

    reg = list_registered()
    ops = sorted({k[0] for k in reg})
    lines = ["operation".ljust(24) + "".join(b.ljust(8) for b in KNOWN_BACKENDS)]
    for op in ops:
        cells = "".join(("yes" if (op, b) in reg else "-").ljust(8) for b in KNOWN_BACKENDS)
        lines.append(op.ljust(24) + cells)
    prov = collect_provenance()
    lines.append("")
    for key in ("econkit", "python", "numpy", "scipy", "numba", "jax"):
        lines.append(f"{key}: {prov[key]}")
    return "\n".join(lines)
