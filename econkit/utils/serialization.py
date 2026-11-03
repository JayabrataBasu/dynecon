"""JSON-safe encoding of results (NumPy types, NaN and infinity encoded explicitly)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["decode", "encode", "numeric_leaves"]

_SPECIAL = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def encode(obj: Any) -> Any:
    """Recursively convert to JSON-compatible objects; NaN/inf become the strings
    ``"NaN"``, ``"Infinity"``, ``"-Infinity"`` and arrays become nested lists."""
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return encode(obj.tolist())
    if isinstance(obj, np.generic):
        return encode(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    return repr(obj)  # objects that cannot be serialised (never code) are recorded by name


def decode(obj: Any) -> Any:
    """Inverse of :func:`encode` for the special float strings (lists stay lists)."""
    if isinstance(obj, dict):
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    if isinstance(obj, str) and obj in _SPECIAL:
        return _SPECIAL[obj]
    return obj


def numeric_leaves(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten nested dicts/lists to ``{path: float}`` for numeric leaves (for comparisons)."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(numeric_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out
