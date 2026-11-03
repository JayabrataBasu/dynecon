"""Impulse responses, theoretical moments and linear simulation of perturbation solutions.

Economics note:
Once a model is solved to first order its dynamics are a linear VAR(1) in the states, so
impulse responses and the full second-moment structure are available in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any  # noqa: F401  (resolves annotations inherited from Solution for the docs)

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.core.solution import Solution
from econkit.model.compiled import CompiledModel  # noqa: F401  (resolves inherited annotations)
from econkit.simulation.result import SimulationResult
from econkit.utils.rng import as_generator

__all__ = [
    "PerturbationSolution",
    "TheoreticalMoments",
    "irf",
    "simulate_linear",
    "theoretical_moments",
]

Array = npt.NDArray[np.float64]


@dataclass(kw_only=True)
class PerturbationSolution(Solution):
    """First-order solution: ``u_t = F x_t``, ``x_{t+1} = P x_t + Sigma e_{t+1}`` (deviations).

    ``policy`` holds ``"F"``, ``"P"``, ``"Sigma"``; deviations are log-deviations for variables
    in ``log_variables`` and level deviations otherwise.
    """

    variables: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    shocks: tuple[str, ...] = ()
    log_variables: tuple[str, ...] = ()
    steady_state: Array = field(default_factory=lambda: np.zeros(0))

    @property
    def F(self) -> Array:
        return np.asarray(self.policy["F"])

    @property
    def P(self) -> Array:
        return np.asarray(self.policy["P"])

    @property
    def Sigma(self) -> Array:
        return np.asarray(self.policy["Sigma"])

    @property
    def observation_matrix(self) -> Array:
        """``C`` with ``y_hat = C x_t`` for all variables (states first)."""
        n_x = len(self.states)
        return np.vstack([np.eye(n_x), self.F])


def _shock_index(sol: PerturbationSolution, shock: str | int) -> int:
    if isinstance(shock, int):
        if not 0 <= shock < len(sol.shocks):
            raise ValidationError(f"shock index {shock} out of range")
        return shock
    if shock not in sol.shocks:
        raise ValidationError(f"unknown shock {shock!r}; known: {list(sol.shocks)}")
    return sol.shocks.index(shock)


def irf(
    solution: PerturbationSolution, shock: str | int, size: float = 1.0, horizon: int = 40
) -> dict[str, Array]:
    """Impulse response to a ``size``-standard-deviation shock at ``t = 0``.

    ``x_0 = Sigma e`` (the shock hits the states at ``t = 0``); responses are deviations from
    steady state, one array of length ``horizon`` per variable.
    """
    if horizon < 1:
        raise ValidationError("horizon must be >= 1")
    k = _shock_index(solution, shock)
    x = solution.Sigma[:, k] * size
    C = solution.observation_matrix
    out = np.empty((horizon, len(solution.variables)))
    for t in range(horizon):
        out[t] = C @ x
        x = solution.P @ x
    return {name: out[:, i] for i, name in enumerate(solution.variables)}


@dataclass(frozen=True)
class TheoreticalMoments:
    """Unconditional second moments of all variables (deviations)."""

    variables: tuple[str, ...]
    covariance: Array
    autocovariances: dict[int, Array]
    variance_decomposition: Array

    @property
    def std(self) -> Array:
        return np.sqrt(np.diag(self.covariance))

    def autocorrelation(self, lag: int) -> Array:
        """Own autocorrelation of each variable at ``lag``."""
        return np.diag(self.autocovariances[lag]) / np.diag(self.covariance)


def theoretical_moments(
    solution: PerturbationSolution, lags: tuple[int, ...] = (1,)
) -> TheoreticalMoments:
    """Covariance, autocovariances and shock variance decomposition via the Lyapunov equation."""
    from scipy.linalg import solve_discrete_lyapunov

    P, S, C = solution.P, solution.Sigma, solution.observation_matrix
    gamma0 = solve_discrete_lyapunov(P, S @ S.T)
    cov = C @ gamma0 @ C.T
    autocov = {k: C @ np.linalg.matrix_power(P, k) @ gamma0 @ C.T for k in lags}
    n_e = S.shape[1]
    contrib = np.empty((C.shape[0], n_e))
    for e in range(n_e):
        g = solve_discrete_lyapunov(P, np.outer(S[:, e], S[:, e]))
        contrib[:, e] = np.diag(C @ g @ C.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = contrib / contrib.sum(axis=1, keepdims=True)
    return TheoreticalMoments(solution.variables, cov, autocov, np.nan_to_num(share))


def simulate_linear(
    solution: PerturbationSolution,
    *,
    n_periods: int,
    n_agents: int = 1,
    burn_in: int = 0,
    rng: np.random.Generator | int | None = None,
    store: list[str] | None = None,
) -> SimulationResult:
    """Simulate the linear law of motion; paths are deviations from steady state.

    The initial state is drawn from the stationary distribution ``N(0, Gamma_0)``.
    """
    from scipy.linalg import solve_discrete_lyapunov

    if n_periods < 1 or n_agents < 1 or burn_in < 0:
        raise ValidationError("need n_periods >= 1, n_agents >= 1, burn_in >= 0")
    gen = as_generator(rng)
    P, S = solution.P, solution.Sigma
    n_x = P.shape[0]
    total = burn_in + n_periods
    gamma0 = solve_discrete_lyapunov(P, S @ S.T)
    w, v = np.linalg.eigh((gamma0 + gamma0.T) / 2.0)
    root = v * np.sqrt(np.clip(w, 0.0, None))
    x = gen.standard_normal((n_agents, n_x)) @ root.T
    eps = gen.standard_normal((total, n_agents, S.shape[1]))
    xs = np.empty((total, n_agents, n_x))
    for t in range(total):
        xs[t] = x
        x = x @ P.T + eps[t] @ S.T
    y = xs @ solution.observation_matrix.T  # (total, n_agents, n_y)
    paths = {
        name: np.ascontiguousarray(y[burn_in:, :, i]) for i, name in enumerate(solution.variables)
    }
    if store is not None:
        missing = [s for s in store if s not in paths]
        if missing:
            raise ValidationError(f"cannot store unknown variables {missing}")
        paths = {k: v for k, v in paths.items() if k in store}
    entropy = getattr(gen.bit_generator.seed_seq, "entropy", None)  # type: ignore[attr-defined]
    return SimulationResult(
        paths=paths,
        n_periods=n_periods,
        n_agents=n_agents,
        seed_entropy=int(entropy) if isinstance(entropy, int) else None,
        solution_fingerprint=str(solution.metadata.get("fingerprint", "")),
        metadata={"backend": "numpy", "burn_in": burn_in, "units": "deviations"},
    )
