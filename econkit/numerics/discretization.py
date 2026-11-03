"""Discretisation of AR(1) processes into finite Markov chains (cold path, NumPy/SciPy).

Economics note:
Maps a continuous income/productivity process into the finite Markov chain used by
dynamic-programming solvers. Rouwenhorst stays accurate for persistent processes.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError

__all__ = ["rouwenhorst", "tauchen"]

Array = npt.NDArray[np.float64]


def _check(rho: float, sigma_e: float, n_states: int) -> None:
    if int(n_states) != n_states or n_states < 2:
        raise ValidationError(f"n_states must be an integer >= 2, got {n_states}")
    if not -1.0 < rho < 1.0:
        raise ValidationError(f"rho must lie in (-1, 1), got {rho}")
    if not sigma_e > 0.0:
        raise ValidationError(f"sigma_e must be positive, got {sigma_e}")


def tauchen(rho: float, sigma_e: float, n_states: int, m: float = 3.0) -> tuple[Array, Array]:
    """Tauchen (1986) grid and transition matrix for ``z' = rho z + eps``, ``eps ~ N(0, s^2)``."""
    from scipy.special import ndtr

    _check(rho, sigma_e, n_states)
    sigma_z = sigma_e / math.sqrt(1.0 - rho * rho)
    grid = np.linspace(-m * sigma_z, m * sigma_z, n_states, dtype=np.float64)
    step = grid[1] - grid[0]
    mean = rho * grid[:, None]
    upper = (grid[None, :] - mean + 0.5 * step) / sigma_e
    lower = (grid[None, :] - mean - 0.5 * step) / sigma_e
    trans = ndtr(upper) - ndtr(lower)
    trans[:, 0] = ndtr(upper[:, 0])
    trans[:, -1] = 1.0 - ndtr(lower[:, -1])
    trans /= trans.sum(axis=1, keepdims=True)
    return grid, trans


def rouwenhorst(rho: float, sigma_e: float, n_states: int) -> tuple[Array, Array]:
    """Rouwenhorst (1995) grid and transition matrix; matches the AR(1) variance and persistence."""
    _check(rho, sigma_e, n_states)
    p = (1.0 + rho) / 2.0
    trans = np.array([[p, 1.0 - p], [1.0 - p, p]], dtype=np.float64)
    for n in range(3, n_states + 1):
        prev = trans
        trans = np.zeros((n, n), dtype=np.float64)
        trans[:-1, :-1] += p * prev
        trans[:-1, 1:] += (1.0 - p) * prev
        trans[1:, :-1] += (1.0 - p) * prev
        trans[1:, 1:] += p * prev
        trans[1:-1, :] /= 2.0
    psi = sigma_e / math.sqrt(1.0 - rho * rho) * math.sqrt(n_states - 1)
    grid = np.linspace(-psi, psi, n_states, dtype=np.float64)
    trans /= trans.sum(axis=1, keepdims=True)
    return grid, trans
