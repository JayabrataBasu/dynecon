"""The one fixed-point iteration driver used by every solver.

Economics note:
Value function iteration, EGM, the Lucas-tree price functional and stationary
distribution iteration are all contraction mappings; sharing one loop guarantees
identical convergence semantics (and warnings) across the library.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from econkit.config import config
from econkit.core.errors import ConvergenceError, ConvergenceWarning, ValidationError

__all__ = ["FixedPointResult", "fixed_point"]

Array = npt.NDArray[np.float64]
Callback = Callable[[int, Array, float], bool | None]


@dataclass(frozen=True)
class FixedPointResult:
    """Outcome of :func:`fixed_point`."""

    x: Array
    converged: bool
    iterations: int
    residual: float
    history: tuple[float, ...] = field(default_factory=tuple)


def _distance(a: Array, b: Array, norm: str) -> float:
    d = np.abs(a - b)
    return float(d.max()) if norm == "sup" else float(d.sum())


def fixed_point(
    operator: Callable[[Array], Array],
    x0: Array,
    *,
    tol: float,
    max_iter: int,
    norm: str = "sup",
    damping: float = 1.0,
    callbacks: Sequence[Callback] = (),
    name: str = "fixed_point",
) -> FixedPointResult:
    """Iterate ``x <- damping * T(x) + (1 - damping) * x`` until ``||T(x) - x|| < tol``.

    ``norm`` is ``"sup"`` or ``"l1"``. Callbacks receive ``(iteration, x, residual)`` and may
    return ``True`` to stop early. Non-convergence emits :class:`ConvergenceWarning`
    (or raises :class:`ConvergenceError` when ``config.strict_convergence``).
    """
    if not tol > 0.0:
        raise ValidationError("tol must be strictly positive")
    if max_iter < 1:
        raise ValidationError("max_iter must be >= 1")
    if not 0.0 < damping <= 1.0:
        raise ValidationError("damping must lie in (0, 1]")
    if norm not in ("sup", "l1"):
        raise ValidationError("norm must be 'sup' or 'l1'")

    x = np.array(x0, dtype=np.float64, copy=True)
    history: list[float] = []
    residual = float("inf")
    converged = False
    stopped = False
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007 - used after the loop
        tx = np.asarray(operator(x), dtype=np.float64)
        if tx.shape != x.shape:
            raise ValidationError(f"{name}: operator returned shape {tx.shape}, expected {x.shape}")
        residual = _distance(tx, x, norm)
        history.append(residual)
        x = tx if damping == 1.0 else damping * tx + (1.0 - damping) * x
        if residual < tol:
            converged = True
            break
        if any(cb(iterations, x, residual) for cb in callbacks):
            stopped = True
            break
    if not converged and not stopped:
        msg = f"{name} did not converge in {max_iter} iterations (residual {residual:.3e})"
        if config.strict_convergence:
            raise ConvergenceError(msg)
        warnings.warn(msg, ConvergenceWarning, stacklevel=2)
    return FixedPointResult(x, converged, iterations, residual, tuple(history))
