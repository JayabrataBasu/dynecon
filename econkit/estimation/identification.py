"""Identification diagnostics for SMM estimates.

Economics note:
A parameter is identified when moving it changes some matched moment. If two parameters
only matter through their product, the data cannot tell them apart; the moment Jacobian's
rank and condition number reveal this before standard errors blow up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from econkit.core.errors import IdentificationError, ValidationError
from econkit.estimation.results import SMMResult

__all__ = ["IdentificationReport", "identification_report", "loss_profile"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class IdentificationReport:
    """Rank, conditioning and weakly identified parameters of the moment Jacobian."""

    rank: int
    condition_number: float
    singular_values: Array
    weak_parameters: tuple[str, ...]
    names: tuple[str, ...]

    def summary(self) -> str:
        lines = [
            f"rank {self.rank} of {len(self.names)} parameters, "
            f"cond(G'WG) = {self.condition_number:.3g}",
            "singular values (scaled Jacobian): "
            + ", ".join(f"{s:.3g}" for s in self.singular_values),
        ]
        if self.weak_parameters:
            lines.append("weakly identified: " + ", ".join(self.weak_parameters))
        return "\n".join(lines)


def identification_report(
    result: SMMResult, weak_tol: float = 1e-3, rank_tol: float = 1e-6
) -> IdentificationReport:
    """Diagnose identification from the Jacobian ``G`` of an SMM result.

    Singular values are computed on ``G`` with columns scaled by ``|theta|`` (elasticities),
    so parameters on different scales are comparable. Raises :class:`IdentificationError` if
    the scaled Jacobian is rank-deficient (relative singular value below ``rank_tol``).
    """
    scale = np.maximum(np.abs(result.estimates), 1e-8)
    Gs = result.jacobian * scale
    u, s, vt = np.linalg.svd(Gs, full_matrices=False)
    rel = s / s[0] if s[0] > 0 else np.zeros_like(s)
    rank = int(np.sum(rel > rank_tol))
    gwg = result.jacobian.T @ result.weights @ result.jacobian
    cond = float(np.linalg.cond(gwg)) if np.all(np.isfinite(gwg)) else float("inf")
    weak_idx = np.flatnonzero(rel < weak_tol)
    flagged: list[str] = []
    for i, name in enumerate(result.names):
        if weak_idx.size and float(np.sum(vt[weak_idx, i] ** 2)) > 0.5:
            flagged.append(name)
    report = IdentificationReport(rank, cond, s, tuple(flagged), tuple(result.names))
    if rank < len(result.names):
        raise IdentificationError(
            f"moment Jacobian is rank-deficient (rank {rank} < {len(result.names)} parameters); "
            f"weakly identified: {flagged or result.names}"
        )
    return report


def loss_profile(
    result: SMMResult, name: str, grid: Sequence[float], reoptimize: bool = False
) -> Array:
    """SMM loss along ``name`` over ``grid``; with ``reoptimize`` the other parameters re-fit."""
    if name not in result.names:
        raise ValidationError(f"unknown parameter {name!r}; have {result.names}")
    others = [n for n in result.names if n != name]
    out = np.empty(len(grid))
    for k, value in enumerate(grid):
        base = dict(result.params)
        base[name] = float(value)
        if not (reoptimize and others):
            out[k] = result.loss_fn(base)
            continue

        def obj(x: Array) -> float:
            trial = {**base, **dict(zip(others, x, strict=True))}
            return result.loss_fn(trial)

        start = np.array([base[n] for n in others])
        out[k] = float(minimize(obj, start, method="Nelder-Mead").fun)
    return out
