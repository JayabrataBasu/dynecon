"""Linear-quadratic control and the discrete algebraic Riccati equation.

Economics note:
Many macro problems (permanent-income consumption, Ramsey taxation, optimal monetary policy
around a steady state) have quadratic loss and linear dynamics. Their optimal policy is linear,
``u = -F x``, and the value is quadratic, ``x'Px + d``, so a Riccati recursion replaces a whole
dynamic program.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_discrete_are

from dynecon.core.errors import ValidationError
from dynecon.utils.rng import as_generator

__all__ = ["LQ", "solve_discrete_riccati"]

Array = npt.NDArray[np.float64]


def _mat(x: npt.ArrayLike, name: str, shape: tuple[int, int] | None = None) -> Array:
    a = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValidationError(f"{name} must be a finite matrix")
    if shape is not None and a.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}, got {a.shape}")
    return a


def solve_discrete_riccati(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    R: npt.ArrayLike,
    Q: npt.ArrayLike,
    N: npt.ArrayLike | None = None,
    beta: float = 1.0,
) -> Array:
    """Solve ``P = R + beta A'PA - (beta A'PB + N')(Q + beta B'PB)^-1 (beta B'PA + N)``.

    This is the value-matrix equation of ``min E sum beta^t (x'Rx + u'Qu + 2u'Nx)`` subject to
    ``x' = Ax + Bu + Cw``. ``N`` has shape ``(k, n)`` and defaults to zero.
    """
    A, B = _mat(A, "A"), _mat(B, "B")
    n, k = A.shape[0], B.shape[1]
    if A.shape != (n, n) or B.shape[0] != n:
        raise ValidationError("A must be square and B must have as many rows as A")
    R, Q = _mat(R, "R", (n, n)), _mat(Q, "Q", (k, k))
    Nm = np.zeros((k, n)) if N is None else _mat(N, "N", (k, n))
    if not 0.0 < beta <= 1.0:
        raise ValidationError("beta must lie in (0, 1]")
    s = np.sqrt(beta)  # discounting is undone by scaling the dynamics (x~ = beta^(t/2) x)
    try:
        P = solve_discrete_are(s * A, s * B, R, Q, s=Nm.T)
    except (np.linalg.LinAlgError, ValueError) as err:
        raise ValidationError(f"the Riccati equation has no stabilising solution: {err}") from err
    return np.asarray(P, dtype=np.float64)


class LQ:
    """Linear-quadratic regulator: ``min E sum beta^t (x'Rx + u'Qu + 2u'Nx)``.

    The state follows ``x' = Ax + Bu + Cw``.

    This class **minimises** cost. With ``T=None`` the horizon is infinite and
    :meth:`stationary_values` returns ``(P, F, d)``: the policy is ``u = -F x`` and the cost is
    ``x'Px + d``. With a finite ``T`` and terminal cost ``x'Rf x``, :meth:`update_values` runs the
    backward recursion and :attr:`policies` holds ``F_0, ..., F_{T-1}``.
    """

    def __init__(
        self,
        Q: npt.ArrayLike,
        R: npt.ArrayLike,
        A: npt.ArrayLike,
        B: npt.ArrayLike,
        C: npt.ArrayLike | None = None,
        N: npt.ArrayLike | None = None,
        beta: float = 1.0,
        T: int | None = None,
        Rf: npt.ArrayLike | None = None,
    ) -> None:
        self.A, self.B = _mat(A, "A"), _mat(B, "B")
        self.n, self.k = self.A.shape[0], self.B.shape[1]
        if self.A.shape != (self.n, self.n) or self.B.shape[0] != self.n:
            raise ValidationError("A must be square and B must have as many rows as A")
        self.R, self.Q = _mat(R, "R", (self.n, self.n)), _mat(Q, "Q", (self.k, self.k))
        self.N = np.zeros((self.k, self.n)) if N is None else _mat(N, "N", (self.k, self.n))
        self.C = np.zeros((self.n, 1)) if C is None else _mat(C, "C")
        if self.C.shape[0] != self.n:
            raise ValidationError("C must have as many rows as A")
        if not 0.0 < beta <= 1.0:
            raise ValidationError("beta must lie in (0, 1]")
        if T is not None and (int(T) != T or T < 1):
            raise ValidationError("T must be a positive integer or None")
        if T is None and beta == 1.0 and np.any(self.C != 0):
            raise ValidationError("beta < 1 is needed when there is noise and an infinite horizon")
        self.beta, self.T = float(beta), T
        self.Rf = self.R if Rf is None else _mat(Rf, "Rf", (self.n, self.n))
        self.P, self.d, self.F = self.Rf.copy(), 0.0, np.zeros((self.k, self.n))
        self.policies: list[Array] = []

    def _step(self, P: Array, d: float) -> tuple[Array, Array, float]:
        """One Riccati step: next ``P``, the policy ``F`` for the current ``P``, and ``d``."""
        b = self.beta
        S = self.Q + b * self.B.T @ P @ self.B
        F = np.linalg.solve(S, b * self.B.T @ P @ self.A + self.N)
        P_new = self.R + b * self.A.T @ P @ self.A - (b * self.A.T @ P @ self.B + self.N.T) @ F
        d_new = b * (d + float(np.trace(P @ self.C @ self.C.T)))
        return 0.5 * (P_new + P_new.T), F, d_new

    def stationary_values(self) -> tuple[Array, Array, float]:
        """Infinite-horizon ``(P, F, d)``; also stored on the object."""
        P = solve_discrete_riccati(self.A, self.B, self.R, self.Q, self.N, self.beta)
        F = np.linalg.solve(
            self.Q + self.beta * self.B.T @ P @ self.B,
            self.beta * self.B.T @ P @ self.A + self.N,
        )
        d = (
            self.beta * float(np.trace(P @ self.C @ self.C.T)) / (1.0 - self.beta)
            if self.beta < 1
            else 0.0
        )
        self.P, self.F, self.d = P, F, d
        return P, F, d

    def update_values(self) -> None:
        """Backward recursion over the finite horizon, from the terminal cost ``Rf``."""
        if self.T is None:
            raise ValidationError("update_values needs a finite horizon T")
        P, d, policies = self.Rf.copy(), 0.0, []
        for _ in range(self.T):
            P, F, d = self._step(P, d)
            policies.append(F)
        self.P, self.d, self.F = P, d, policies[-1]
        self.policies = policies[::-1]  # policies[t] is the policy applied at time t

    def compute_sequence(
        self,
        x0: npt.ArrayLike,
        ts_length: int | None = None,
        w: npt.ArrayLike | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> tuple[Array, Array, Array]:
        """Simulate ``(x, u, w)`` under the optimal policy from ``x0``.

        Noise ``w`` has shape ``(ts_length, shocks)`` (drawn standard normal when omitted); the
        state is updated as ``x_{t+1} = A x_t + B u_t + C w_{t+1}``.
        """
        x = np.asarray(x0, dtype=np.float64).ravel()
        if x.size != self.n:
            raise ValidationError(f"x0 must have {self.n} entries")
        if self.T is None:
            if ts_length is None or ts_length < 1:
                raise ValidationError("ts_length >= 1 is needed for an infinite horizon")
            F_path = [self.F if self.F.any() else self.stationary_values()[1]] * ts_length
        else:
            if not self.policies:
                self.update_values()
            F_path = self.policies
            ts_length = len(F_path)
        m = self.C.shape[1]
        noise = (
            as_generator(rng).standard_normal((ts_length, m))
            if w is None
            else np.asarray(w, dtype=np.float64).reshape(ts_length, m)
        )
        xs, us = np.empty((ts_length + 1, self.n)), np.empty((ts_length, self.k))
        xs[0] = x
        for t in range(ts_length):
            us[t] = -F_path[t] @ xs[t]
            xs[t + 1] = self.A @ xs[t] + self.B @ us[t] + self.C @ noise[t]
        return xs, us, noise
