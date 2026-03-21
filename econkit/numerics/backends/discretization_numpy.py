"""NumPy reference discretization kernels for AR(1) shock processes."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def tauchen_numpy(
    rho: float,
    sigma_e: float,
    n_states: int,
    m: float = 3.0,
) -> tuple[Array, Array]:
    """Tauchen discretization for AR(1): z' = rho * z + eps.

    Economics note:
    This maps a continuous income/productivity process into a finite Markov
    chain used by dynamic programming solvers.
    """
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie in (-1, 1)")
    if sigma_e <= 0.0:
        raise ValueError("sigma_e must be positive")

    sigma_z = sigma_e / math.sqrt(1.0 - rho * rho)
    z_max = m * sigma_z
    z_min = -z_max
    grid = np.linspace(z_min, z_max, n_states, dtype=np.float64)
    step = grid[1] - grid[0]

    transition = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(n_states):
            if j == 0:
                upper = (grid[0] - rho * grid[i] + 0.5 * step) / sigma_e
                transition[i, j] = _std_norm_cdf(upper)
            elif j == n_states - 1:
                lower = (grid[-1] - rho * grid[i] - 0.5 * step) / sigma_e
                transition[i, j] = 1.0 - _std_norm_cdf(lower)
            else:
                upper = (grid[j] - rho * grid[i] + 0.5 * step) / sigma_e
                lower = (grid[j] - rho * grid[i] - 0.5 * step) / sigma_e
                transition[i, j] = _std_norm_cdf(upper) - _std_norm_cdf(lower)

    transition /= transition.sum(axis=1, keepdims=True)
    return grid, transition


def rouwenhorst_numpy(
    rho: float,
    sigma_e: float,
    n_states: int,
) -> tuple[Array, Array]:
    """Rouwenhorst discretization for persistent AR(1) shock processes."""
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie in (-1, 1)")
    if sigma_e <= 0.0:
        raise ValueError("sigma_e must be positive")

    p = (1.0 + rho) / 2.0
    q = p
    transition = np.array([[p, 1.0 - p], [1.0 - q, q]], dtype=np.float64)

    for n in range(3, n_states + 1):
        prev = transition
        transition = np.zeros((n, n), dtype=np.float64)
        transition[:-1, :-1] += p * prev
        transition[:-1, 1:] += (1.0 - p) * prev
        transition[1:, :-1] += (1.0 - q) * prev
        transition[1:, 1:] += q * prev
        transition[1:-1, :] /= 2.0

    sigma_z = sigma_e / math.sqrt(1.0 - rho * rho)
    psi = sigma_z * math.sqrt(n_states - 1)
    grid = np.linspace(-psi, psi, n_states, dtype=np.float64)
    transition /= transition.sum(axis=1, keepdims=True)
    return grid, transition
