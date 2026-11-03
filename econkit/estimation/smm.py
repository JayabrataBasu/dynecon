"""Simulated method of moments.

Economics note:
SMM picks the structural parameters that make the model's simulated moments closest to
the data's. With as many moments as parameters the minimum distance is zero and the model
hits the targets exactly (calibration); with more moments the J-test checks over-identification.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import minimize

from econkit.core.errors import (
    ConvergenceError,
    EconKitError,
    ValidationError,
)
from econkit.estimation.moments import MomentSet
from econkit.estimation.results import SMMResult
from econkit.model.compiled import CompiledModel
from econkit.simulation.simulate import simulate
from econkit.solvers.api import solve
from econkit.utils.rng import as_generator

__all__ = ["SMMConfig", "make_moment_fn", "smm"]

Array = npt.NDArray[np.float64]
MomentFn = Callable[[CompiledModel, int], Array]


@dataclass(frozen=True)
class SMMConfig:
    """Frozen record of the SMM options (stored on the result for reproducibility)."""

    method: str = "Nelder-Mead"
    weights: str = "identity"
    n_obs: int = 1
    n_sim: float = 1.0
    seed: int = 0
    omega_reps: int = 40
    fd_step: float = 1e-3
    max_iter: int = 2000
    penalty: float = 1e10


def make_moment_fn(
    moments: MomentSet,
    solve_options: dict[str, Any] | None = None,
    sim_options: dict[str, Any] | None = None,
) -> MomentFn:
    """Default pipeline ``solve -> simulate -> moments`` for use as an SMM ``moment_fn``."""
    sopts, mopts = dict(solve_options or {}), dict(sim_options or {})

    def fn(model: CompiledModel, seed: int) -> Array:
        sim = simulate(solve(model, **sopts), rng=seed, **mopts)
        return moments.model_moments(sim)

    return fn


def _check_weights(weights: str | Array, q: int) -> tuple[str, Array | None]:
    if isinstance(weights, str):
        if weights not in ("identity", "optimal"):
            raise ValidationError("weights must be 'identity', 'optimal' or a matrix")
        return weights, None
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (q, q):
        raise ValidationError(f"weight matrix must have shape {(q, q)}, got {w.shape}")
    return "custom", w


def smm(  # noqa: C901 - sequential two-step estimation with inference; steps are inline
    model: CompiledModel,
    free_params: Sequence[str],
    data_moments: npt.ArrayLike | None,
    moment_fn: MomentFn | MomentSet,
    *,
    n_obs: int,
    n_sim: float = 1.0,
    solve_options: dict[str, Any] | None = None,
    sim_options: dict[str, Any] | None = None,
    weights: str | Array = "identity",
    omega: Array | None = None,
    method: str = "Nelder-Mead",
    rng: np.random.Generator | int | None = 0,
    omega_reps: int = 40,
    fd_step: float = 1e-3,
    max_iter: int = 2000,
    penalty: float = 1e10,
) -> SMMResult:
    """Estimate ``free_params`` by matching simulated to data moments.

    ``moment_fn(model_at_theta, seed)`` returns the model-implied moment vector; the same
    ``seed`` is used at every evaluation (common random numbers). A :class:`MomentSet` may be
    passed instead, in which case ``solve -> simulate -> moments`` is used with
    ``solve_options`` / ``sim_options``. ``n_sim`` is the ratio of simulated to data sample
    size and ``n_obs`` the data sample size. Weighting is ``"identity"``, a matrix, or
    ``"optimal"`` (two-step with ``Omega`` from ``omega`` or simulation at the first-step
    estimate). Standard errors follow ``(1 + 1/S)(G'WG)^-1 G'W Omega W G (G'WG)^-1 / N``.
    """
    names = list(free_params)
    if not names:
        raise ValidationError("free_params must name at least one parameter")
    if isinstance(moment_fn, MomentSet):
        if data_moments is None:
            data_moments = moment_fn.data
        moment_fn = make_moment_fn(moment_fn, solve_options, sim_options)
    m_hat = np.asarray(data_moments, dtype=np.float64).ravel()
    q, p = m_hat.size, len(names)
    if q < p:
        raise ValidationError(f"need at least as many moments ({q}) as parameters ({p})")
    wkind, w_given = _check_weights(weights, q)
    seed = int(as_generator(rng).integers(2**31 - 1000))
    ps = model.parameters
    theta0 = ps.vector(names)
    u0 = ps.to_unconstrained(theta0, names)
    counters = {"evals": 0, "failed": 0}

    def moments_at(theta: Array, s: int) -> Array | None:
        try:
            m = np.asarray(moment_fn(model.with_params(**dict(zip(names, theta, strict=True))), s))
        except (EconKitError, FloatingPointError, ValueError, ArithmeticError):
            return None
        return m if m.shape == (q,) and np.all(np.isfinite(m)) else None

    def loss_theta(theta: Array, W: Array) -> float:
        counters["evals"] += 1
        m = moments_at(theta, seed)
        if m is None:
            counters["failed"] += 1
            return penalty
        g = m - m_hat
        return float(g @ W @ g)

    def fit(W: Array, start: Array) -> Any:
        def obj(u: Array) -> float:
            return loss_theta(ps.from_unconstrained(u, names), W)

        opts: dict[str, Any] = {"maxiter": max_iter}
        if method == "Nelder-Mead":
            opts.update(maxfev=max_iter, xatol=1e-7, fatol=1e-14)
        return minimize(obj, start, method=method, options=opts)

    W1 = np.eye(q) if w_given is None else w_given
    res = fit(W1, u0)
    W_final = W1

    def estimate_omega(theta: Array) -> Array:
        reps = []
        for r in range(omega_reps):
            m = moments_at(theta, seed + 1 + r)
            if m is not None:
                reps.append(m)
        if len(reps) < max(5, p + 1):
            raise ConvergenceError("too few successful simulations to estimate Omega")
        return np.atleast_2d(np.cov(np.array(reps).T)) * (n_obs * n_sim)

    theta1 = ps.from_unconstrained(res.x, names)
    om = omega if omega is not None else None
    if wkind == "optimal":
        om = estimate_omega(theta1) if om is None else om
        W_final = np.linalg.pinv(om)
        res = fit(W_final, res.x)
    theta = ps.from_unconstrained(res.x, names)
    if om is None:
        om = estimate_omega(theta)

    m_model = moments_at(theta, seed)
    if m_model is None:
        raise ConvergenceError("model moments are undefined at the estimate")
    lo_hi = ps.bounds(names)

    def jacobian(rel: float) -> Array:
        G = np.empty((q, p))
        for i in range(p):
            h = rel * max(abs(theta[i]), 1e-3)
            up = min(theta[i] + h, lo_hi[i][1] - 1e-12 * (1 + abs(theta[i])))
            dn = max(theta[i] - h, lo_hi[i][0] + 1e-12 * (1 + abs(theta[i])))
            tp, tm = theta.copy(), theta.copy()
            tp[i], tm[i] = up, dn
            mp, mm = moments_at(tp, seed), moments_at(tm, seed)
            if mp is None or mm is None:
                raise ConvergenceError("moments undefined in the Jacobian step")
            G[:, i] = (mp - mm) / (up - dn)
        return G

    G = jacobian(fd_step)
    G2 = jacobian(2 * fd_step)
    step_gap = float(np.max(np.abs(G - G2) / (np.abs(G) + 1e-8)))
    bread = np.linalg.pinv(G.T @ W_final @ G)
    cov = (1 + 1 / n_sim) * bread @ G.T @ W_final @ om @ W_final @ G @ bread / n_obs
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    g = m_model - m_hat
    df = q - p
    j_stat = j_p = None
    if df > 0:
        j_stat = float(n_obs / (1 + 1 / n_sim) * g @ np.linalg.pinv(om) @ g)
        j_p = float(stats.chi2.sf(j_stat, df))

    def loss_fn(params: dict[str, float]) -> float:
        return loss_theta(np.array([params[n] for n in names]), W_final)

    cfg = SMMConfig(
        method=method,
        weights=wkind,
        n_obs=n_obs,
        n_sim=n_sim,
        seed=seed,
        omega_reps=omega_reps,
        fd_step=fd_step,
        max_iter=max_iter,
        penalty=penalty,
    )
    return SMMResult(
        names=names,
        estimates=theta,
        std_errors=se,
        loss=float(g @ W_final @ g),
        weights=W_final,
        omega=om,
        jacobian=G,
        model_moments=m_model,
        data_moments=m_hat,
        n_obs=n_obs,
        n_sim=n_sim,
        j_stat=j_stat,
        j_pvalue=j_p,
        df=df,
        converged=bool(res.success),
        diagnostics={
            "failed_evals": counters["failed"],
            "n_evals": counters["evals"],
            "jacobian_step_gap": step_gap,
        },
        config=asdict(cfg),
        loss_fn=loss_fn,
    )
