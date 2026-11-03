"""SHA-256 fingerprint of a compiled model.

Economics note:
A fingerprint ties every solution and simulation to the exact model (parameters,
grids, shocks and equations) that produced it, so changed assumptions are detected.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from econkit.model.problems import (
    DSGEProblem,
    FiniteDPProblem,
    PricingProblem,
    SavingProblem,
)
from econkit.model.specs import ShockProcess

__all__ = ["fingerprint_model"]


def _callable_text(fn: Callable[..., Any] | None) -> str | None:
    if fn is None:
        return None
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return getattr(fn, "__qualname__", repr(type(fn)))


def _array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def fingerprint_model(
    name: str,
    params: Mapping[str, float],
    grids: Mapping[str, np.ndarray],
    shocks: Mapping[str, ShockProcess],
    problem: SavingProblem | FiniteDPProblem | DSGEProblem | PricingProblem | None,
) -> str:
    """Hash parameters (exact floats), grids, shock arrays and equation source text."""
    prob: dict[str, Any] = {}
    if problem is not None:
        prob["kind"] = type(problem).__name__
        for key, val in sorted(vars(problem).items()):
            prob[key] = _callable_text(val) if callable(val) else val
    payload = {
        "name": name,
        "params": {k: float(v).hex() for k, v in sorted(params.items())},
        "grids": {k: _array(v) for k, v in sorted(grids.items())},
        "shocks": {k: [_array(s.values), _array(s.P)] for k, s in sorted(shocks.items())},
        "problem": prob,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
