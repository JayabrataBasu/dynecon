"""Kalman filter for linear-Gaussian state-space models and DSGE likelihoods.

Economics note:
A first-order DSGE solution is a linear state-space model; the Kalman filter delivers its
exact Gaussian likelihood, which makes likelihood-based estimation of DSGE models possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.solvers.perturbation.analysis import PerturbationSolution

__all__ = ["KalmanResult", "kalman_filter", "loglik_state_space"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class KalmanResult:
    """Output of :func:`kalman_filter`.

    ``predicted_state[t]`` / ``predicted_cov[t]`` are ``a_{t|t-1}``, ``P_{t|t-1}``;
    ``filtered_state[t]`` is ``a_{t|t}``; ``gains[t]`` the Kalman gain (zeros if unobserved).
    """

    loglik: float
    predicted_state: Array
    predicted_cov: Array
    filtered_state: Array
    gains: Array
    n_obs_used: int


def kalman_filter(
    y: npt.ArrayLike,
    Z: Array,
    H: Array,
    T: Array,
    Q: Array,
    a0: Array,
    P0: Array,
) -> KalmanResult:
    """Filter ``y = Z a + e``, ``a' = T a + eta`` with ``e ~ N(0, H)``, ``eta ~ N(0, Q)``.

    ``y`` has shape ``(n_periods, n_obs)``; NaN entries are treated as missing and skipped
    (the corresponding rows of ``Z`` and ``H`` are dropped for that period).
    """
    obs = np.asarray(y, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs[:, None]
    n_t, n_y = obs.shape
    n_a = T.shape[0]
    if Z.shape != (n_y, n_a) or H.shape != (n_y, n_y) or Q.shape != (n_a, n_a):
        raise ValidationError("inconsistent state-space matrix shapes")
    a, P = np.array(a0, dtype=np.float64), np.array(P0, dtype=np.float64)
    pred_a = np.empty((n_t, n_a))
    pred_P = np.empty((n_t, n_a, n_a))
    filt_a = np.empty((n_t, n_a))
    gains = np.zeros((n_t, n_a, n_y))
    loglik, used = 0.0, 0
    steady: tuple[Array, Array, float] | None = None  # (K, F^-1, log|F|) once P has converged
    const = n_y * np.log(2 * np.pi)
    for t in range(n_t):
        pred_a[t], pred_P[t] = a, P
        idx = np.flatnonzero(~np.isnan(obs[t]))
        if steady is not None and idx.size == n_y:  # time-invariant system, full data: P is fixed
            K, Finv, logdet = steady
            v = obs[t] - Z @ a
            loglik += -0.5 * (const + logdet + v @ Finv @ v)
            used += n_y
            gains[t] = K
            a_f = a + K @ v
            filt_a[t] = a_f
            a = T @ a_f
            continue
        steady = None  # a gap makes P move again; settle detection restarts
        if idx.size:
            Zt, Ht = Z[idx], H[np.ix_(idx, idx)]
            v = obs[t, idx] - Zt @ a
            F = Zt @ P @ Zt.T + Ht
            sign, logdet = np.linalg.slogdet(F)
            if sign <= 0:
                raise ValidationError("innovation covariance is not positive definite")
            Finv = np.asarray(np.linalg.inv(F), dtype=np.float64)
            loglik += -0.5 * (idx.size * np.log(2 * np.pi) + logdet + v @ Finv @ v)
            used += idx.size
            K = P @ Zt.T @ Finv
            gains[t][:, idx] = K
            a_f = a + K @ v
            P_f = P - K @ Zt @ P
        else:
            a_f, P_f = a, P
        filt_a[t] = a_f
        a, P_new = T @ a_f, T @ P_f @ T.T + Q
        P_new = 0.5 * (P_new + P_new.T)
        if idx.size == n_y and np.max(np.abs(P_new - P)) <= 1e-13 * max(1.0, np.max(np.abs(P))):
            steady = (K, Finv, float(logdet))
        P = P_new
    return KalmanResult(float(loglik), pred_a, pred_P, filt_a, gains, used)


def loglik_state_space(
    solution: PerturbationSolution,
    observables: list[str],
    data: npt.ArrayLike,
    measurement_error: float | npt.ArrayLike = 0.0,
) -> float:
    """Gaussian log-likelihood of ``data`` (deviations of ``observables``) under a DSGE solution.

    The state vector is the predetermined block ``x``; observables are rows of ``[I; F]``.
    ``measurement_error`` is the standard deviation (scalar or per observable) of additive
    noise; with fewer shocks than observables it must be positive (stochastic singularity).
    """
    from scipy.linalg import solve_discrete_lyapunov

    for name in observables:
        if name not in solution.variables:
            raise ValidationError(f"unknown observable {name!r}; have {list(solution.variables)}")
    rows = [solution.variables.index(n) for n in observables]
    Z = solution.observation_matrix[rows]
    sd = np.broadcast_to(np.asarray(measurement_error, dtype=np.float64), (len(rows),))
    H = np.diag(sd**2)
    P, S = solution.P, solution.Sigma
    Q = S @ S.T
    P0 = solve_discrete_lyapunov(P, Q)
    return kalman_filter(data, Z, H, P, Q, np.zeros(P.shape[0]), P0).loglik
