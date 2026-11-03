"""Klein (2000) solution of linear rational-expectations models by generalised Schur.

Economics note:
Blanchard-Kahn: a unique stable equilibrium exists iff the number of stable generalised
eigenvalues equals the number of predetermined variables. Too few stable roots means
explosive paths; too many means sunspot (indeterminate) equilibria.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.core.errors import DeterminacyError

__all__ = ["klein_solve"]

Array = npt.NDArray[np.float64]
_STABLE_MARGIN = 1e-9


def _is_stable(alpha: npt.NDArray[np.complex128], beta: npt.NDArray[np.complex128]) -> object:
    """Stable root of ``A y' = B y``: ``|lambda| = |beta / alpha| < 1`` (alpha = 0 is unstable)."""
    return np.abs(beta) < (1.0 - _STABLE_MARGIN) * np.abs(alpha)


def klein_solve(A: Array, B: Array, n_x: int) -> tuple[Array, Array, Array]:
    """Solve ``A E_t[y_{t+1}] = B y_t`` with the first ``n_x`` variables predetermined.

    Returns ``(F, P, eigenvalues)`` with jump variables ``u_t = F x_t`` and the state law of
    motion ``x_{t+1} = P x_t + Sigma e_{t+1}``. Raises :class:`DeterminacyError` when the
    number of stable roots differs from ``n_x``.
    """
    from scipy.linalg import ordqz

    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n) or not 0 <= n_x <= n:
        raise ValueError("A and B must be square of equal size with 0 <= n_x <= n")
    AA, BB, alpha, beta, _, Z = ordqz(A, B, sort=_is_stable, output="complex")
    with np.errstate(divide="ignore", invalid="ignore"):
        eig = np.where(np.abs(alpha) > 0.0, beta / alpha, np.inf)
    n_stable = int(np.sum(np.abs(beta) < (1.0 - _STABLE_MARGIN) * np.abs(alpha)))
    if n_stable > n_x:
        raise DeterminacyError(f"indeterminate: n_stable={n_stable} > n_predetermined={n_x}")
    if n_stable < n_x:
        raise DeterminacyError(
            f"no stable solution (explosive): n_stable={n_stable} < n_predetermined={n_x}"
        )
    n_u = n - n_x
    if n_x == 0:
        return np.zeros((n_u, 0)), np.zeros((0, 0)), np.abs(eig)
    Z11, Z21 = Z[:n_x, :n_x], Z[n_x:, :n_x]
    if np.linalg.cond(Z11) > 1.0 / np.sqrt(np.finfo(float).eps):
        raise DeterminacyError("Z11 is (numerically) singular: the stable subspace is degenerate")
    S11, T11 = AA[:n_x, :n_x], BB[:n_x, :n_x]
    F = Z21 @ np.linalg.inv(Z11)
    P = Z11 @ np.linalg.solve(S11, T11) @ np.linalg.inv(Z11)
    scale = 1.0 + max(np.abs(F).max(initial=0.0), np.abs(P).max(initial=0.0))
    if max(np.abs(F.imag).max(initial=0.0), np.abs(P.imag).max(initial=0.0)) > 1e-8 * scale:
        raise DeterminacyError("solution matrices are complex; check the model equations")
    return np.real(F), np.real(P), np.abs(eig)
