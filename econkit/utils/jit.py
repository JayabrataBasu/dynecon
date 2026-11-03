"""Lazy JIT decorator with a pure-Python fallback.

Economics note:
Hot loops (Bellman maximisation, simulation) are compiled when Numba is present;
without it the identical Python code runs, so results never depend on the
accelerator.

Numba is imported only when a compiled kernel is first *called*, so ``import econkit`` stays
fast and does not load Numba. On the first call every pending kernel is materialised in
definition order and the lazy wrappers are replaced by real Numba dispatchers wherever they
are referenced, so compiled kernels can call each other.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from collections.abc import Callable
from typing import Any

__all__ = ["NUMBA_AVAILABLE", "jit", "prange"]

NUMBA_AVAILABLE: bool = importlib.util.find_spec("numba") is not None

#: Placeholder for ``numba.prange``; replaced inside kernel modules when Numba is loaded.
prange: Any = range

_state: dict[str, Any] = {"numba": None}
_pending: list[_LazyJit] = []


def _disabled() -> bool:
    return os.environ.get("ECONKIT_DISABLE_JIT", "") == "1"


def _numba() -> Any:
    if _state["numba"] is None:
        import numba

        _state["numba"] = numba
    return _state["numba"]


def _fix_prange(func: Callable[..., Any]) -> None:
    g = func.__globals__
    if g.get("prange") is range:
        g["prange"] = _numba().prange


def _rebind(wrapper: Any, dispatcher: Any) -> None:
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith("econkit"):
            continue
        for key, value in list(vars(mod).items()):
            if value is wrapper:
                setattr(mod, key, dispatcher)


class _LazyJit:
    """Callable stand-in for a Numba dispatcher, materialised on first call."""

    def __init__(self, func: Callable[..., Any], options: dict[str, Any]) -> None:
        self.func = func
        self.options = options
        self.dispatcher: Any = None
        functools.update_wrapper(self, func)

    def _build(self) -> Any:
        if self.dispatcher is None:
            _fix_prange(self.func)
            self.dispatcher = _numba().njit(**self.options)(self.func)
            _rebind(self, self.dispatcher)
        return self.dispatcher

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.dispatcher is None:
            for pending in list(_pending):  # definition order: callees before callers
                pending._build()
            _pending.clear()
            self._build()
        return self.dispatcher(*args, **kwargs)


def jit(
    *args: Any,
    parallel: bool = False,
    cache: bool = True,
    fastmath: bool = False,
    **kw: Any,
) -> Any:
    """``numba.njit`` (applied lazily) if Numba is installed and enabled, else the function.

    Usable as ``@jit`` or ``@jit(parallel=True)``. ``fastmath`` defaults to False
    because ``-inf`` payoffs and fastmath do not mix.
    """
    options = {"cache": cache, "parallel": parallel, "fastmath": fastmath, **kw}

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        if not NUMBA_AVAILABLE or _disabled():
            return func
        if _state["numba"] is not None:  # Numba already loaded: compile-on-call, no wrapper
            _fix_prange(func)
            return _state["numba"].njit(**options)(func)  # type: ignore[no-any-return]
        wrapper = _LazyJit(func, options)
        _pending.append(wrapper)
        return wrapper

    if len(args) == 1 and callable(args[0]) and not kw:
        return decorate(args[0])
    return decorate
