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

from dynecon.config import config
from dynecon.core.errors import ConvergenceError, ConvergenceWarning, DyneconError, ValidationError

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


class _Anderson:
    """Safeguarded Anderson acceleration: the plain iteration is always the fallback."""

    def __init__(self, x0: Array, m: int, lower: float | None) -> None:
        self.m, self.lower = m, lower
        self.mem_x: list[Array] = []
        self.mem_g: list[Array] = []
        self.plain = x0  # last plain image T(x) known to be good
        self.accelerated = False
        self.pause = 2 * m  # plain steps first: far from the fixed point T is too nonlinear

    def undo(self) -> Array:
        """Drop the memory, pause acceleration and return to the last good plain image."""
        self.mem_x, self.mem_g, self.accelerated, self.pause = [], [], False, 2 * self.m
        return self.plain

    def step(self, x: Array, tx: Array, residual: float, history: list[float]) -> Array:
        """Next iterate given ``tx = T(x)``; ``history`` already ends with ``residual``."""
        hurt = not residual <= history[-2] if len(history) > 1 else False  # NaN counts as hurt
        low = self.lower is not None and float(np.min(tx)) < self.lower
        if self.accelerated and (hurt or low):
            return self.undo()
        if not np.isfinite(residual):
            raise ConvergenceError("non-finite iterate under acceleration")
        self.plain = tx
        if self.pause:
            self.pause -= 1
            return tx
        g = tx - x
        self.mem_x.append(x.ravel())
        self.mem_g.append(g.ravel())
        del self.mem_x[: -(self.m + 1)], self.mem_g[: -(self.m + 1)]
        self.accelerated = False
        if len(self.mem_x) < 2:
            return tx
        dx = np.diff(np.array(self.mem_x), axis=0).T
        dg = np.diff(np.array(self.mem_g), axis=0).T
        gamma = np.linalg.lstsq(dg, g.ravel(), rcond=None)[0]
        cand = (tx.ravel() - (dx + dg) @ gamma).reshape(x.shape)
        # reject, not clip: clipped iterates are garbage
        if np.isfinite(cand).all() and (self.lower is None or cand.min() >= self.lower):
            self.accelerated = True
            return np.asarray(cand, dtype=np.float64)
        return tx


def _validate(tol: float, max_iter: int, norm: str, damping: float, anderson: int) -> None:
    if not tol > 0.0:
        raise ValidationError("tol must be strictly positive")
    if max_iter < 1:
        raise ValidationError("max_iter must be >= 1")
    if not 0.0 < damping <= 1.0:
        raise ValidationError("damping must lie in (0, 1]")
    if anderson < 0 or (anderson and damping != 1.0):
        raise ValidationError("anderson must be >= 0 and needs damping == 1")
    if norm not in ("sup", "l1"):
        raise ValidationError("norm must be 'sup' or 'l1'")


def fixed_point(
    operator: Callable[[Array], Array],
    x0: Array,
    *,
    tol: float,
    max_iter: int,
    norm: str = "sup",
    damping: float = 1.0,
    anderson: int = 0,
    lower: float | None = None,
    callbacks: Sequence[Callback] = (),
    name: str = "fixed_point",
) -> FixedPointResult:
    """Iterate ``x <- damping * T(x) + (1 - damping) * x`` until ``||T(x) - x|| < tol``.

    ``norm`` is ``"sup"`` or ``"l1"``. Callbacks receive ``(iteration, x, residual)`` and may
    return ``True`` to stop early. ``anderson=m > 0`` extrapolates from the last ``m`` residuals
    (Anderson acceleration; the memory is dropped whenever the residual grows, so the plain
    iteration is the fallback) and accelerated iterates below ``lower`` are rejected.
    Non-convergence emits :class:`ConvergenceWarning` (or raises :class:`ConvergenceError` when
    ``config.strict_convergence``).
    """
    _validate(tol, max_iter, norm, damping, anderson)

    x = np.array(x0, dtype=np.float64, copy=True)
    if x.size == 0:
        raise ValidationError("x0 must not be empty")
    history: list[float] = []
    residual = float("inf")
    converged = False
    stopped = False
    iterations = 0
    acc = _Anderson(x, anderson, lower) if anderson else None
    for iterations in range(1, max_iter + 1):  # noqa: B007 - used after the loop
        try:
            tx = np.asarray(operator(x), dtype=np.float64)
        except DyneconError:
            if acc is None or not acc.accelerated:
                raise
            x = acc.undo()  # the extrapolated iterate left the domain
            continue
        if tx.shape != x.shape:
            raise ValidationError(f"{name}: operator returned shape {tx.shape}, expected {x.shape}")
        residual = _distance(tx, x, norm)
        history.append(residual)
        if residual < tol:
            x = tx
            converged = True
            break
        if acc is not None:
            x = acc.step(x, tx, residual, history)
        else:
            x = tx if damping == 1.0 else damping * tx + (1.0 - damping) * x
        if any(cb(iterations, x, residual) for cb in callbacks):
            stopped = True
            break
    if not converged and not stopped:
        msg = f"{name} did not converge in {max_iter} iterations (residual {residual:.3e})"
        if config.strict_convergence:
            raise ConvergenceError(msg)
        warnings.warn(msg, ConvergenceWarning, stacklevel=2)
    return FixedPointResult(x, converged, iterations, residual, tuple(history))
