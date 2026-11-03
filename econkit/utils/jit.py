"""JIT decorator with a pure-Python fallback.

Economics note:
Hot loops (Bellman maximisation, simulation) are compiled when Numba is present;
without it the identical Python code runs, so results never depend on the
accelerator.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

try:
    import numba as _numba
except ImportError:  # pragma: no cover - optional dependency
    _numba = None  # type: ignore[assignment]

__all__ = ["NUMBA_AVAILABLE", "jit", "prange"]

NUMBA_AVAILABLE: bool = _numba is not None
prange: Any = _numba.prange if _numba is not None else range


def _disabled() -> bool:
    return os.environ.get("ECONKIT_DISABLE_JIT", "") == "1"


def jit(
    *args: Any,
    parallel: bool = False,
    cache: bool = True,
    fastmath: bool = False,
    **kw: Any,
) -> Any:
    """``numba.njit`` if Numba is installed and enabled, else the plain function.

    Usable as ``@jit`` or ``@jit(parallel=True)``. ``fastmath`` defaults to False
    because ``-inf`` payoffs and fastmath do not mix.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        if _numba is None or _disabled():
            return func
        return _numba.njit(  # type: ignore[no-any-return]
            cache=cache, parallel=parallel, fastmath=fastmath, **kw
        )(func)

    if len(args) == 1 and callable(args[0]) and not kw:
        return decorate(args[0])
    return decorate
