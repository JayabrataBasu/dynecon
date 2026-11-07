"""Discretisation of AR(1) processes into finite Markov chains (cold path, NumPy/SciPy).

Economics note:
Maps a continuous income/productivity process into the finite Markov chain used by
dynamic-programming solvers. Rouwenhorst stays accurate for persistent processes.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.utils.validation import check_finite, check_int

__all__ = ["rouwenhorst", "tauchen"]

Array = npt.NDArray[np.float64]
MAX_STATES = 5000  # the transition matrix is dense: n_states ** 2 floats


def _check(rho: float, sigma_e: float, n_states: int) -> tuple[float, float, int]:
    """Validated ``(rho, sigma_e, n_states)``."""
    n = check_int(n_states, "n_states", 2, MAX_STATES)
    rho = check_finite(rho, "rho")
    if not -1.0 < rho < 1.0:
        raise ValidationError(f"rho must lie in (-1, 1), got {rho}")
    return rho, check_finite(sigma_e, "sigma_e", positive=True), n


def _span(half_width: float) -> float:
    if not math.isfinite(half_width):
        raise ValidationError("sigma_e is too large: the state grid overflows")
    return half_width


def tauchen(rho: float, sigma_e: float, n_states: int, m: float = 3.0) -> tuple[Array, Array]:
    """Tauchen (1986) grid and transition matrix for ``z' = rho z + eps``, ``eps ~ N(0, s^2)``."""
    from scipy.special import ndtr

    rho, sigma_e, n_states = _check(rho, sigma_e, n_states)
    m = check_finite(m, "m", positive=True)
    sigma_z = sigma_e / math.sqrt(1.0 - rho * rho)
    half = _span(m * sigma_z)
    grid = np.linspace(-half, half, n_states, dtype=np.float64)
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
    rho, sigma_e, n_states = _check(rho, sigma_e, n_states)
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
    psi = _span(sigma_e / math.sqrt(1.0 - rho * rho) * math.sqrt(n_states - 1))
    grid = np.linspace(-psi, psi, n_states, dtype=np.float64)
    trans /= trans.sum(axis=1, keepdims=True)
    return grid, trans
