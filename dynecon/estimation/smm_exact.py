"""SMM with exact moment Jacobians: moments from the stationary distribution, no simulation noise.

Economics note:
When moments are computed from the model's stationary distribution instead of a simulated
sample, they are smooth deterministic functions of the parameters. Their exact Jacobian gives
a gradient for the optimiser and the matrix ``G`` in the standard errors, so estimation needs
far fewer model solves than a derivative-free search plus finite differences.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.optimize import minimize

from dynecon.core.errors import ConvergenceError, DyneconError, ValidationError
from dynecon.estimation.bayes import ess, split_rhat
from dynecon.estimation.hmc import hmc
from dynecon.estimation.results import MCMCResult, SMMResult
from dynecon.estimation.sensitivity import (
    equilibrium_moments,
    mixture_moments,
    stationary_moments,
)
from dynecon.model.compiled import CompiledModel
from dynecon.solvers.equilibrium import solve_stationary_equilibrium

__all__ = ["quasi_posterior", "smm_exact"]

Array = npt.NDArray[np.float64]
PENALTY = 1e10  # loss reported for parameter values where the model cannot be solved


def _weights(weights: str | Array, q: int) -> tuple[str, Array]:
    if isinstance(weights, str):
        if weights not in ("identity", "optimal"):
            raise ValidationError("weights must be 'identity', 'optimal' or a matrix")
        return weights, np.eye(q)
    return "custom", np.asarray(weights, dtype=np.float64)


def _evaluate(
    model: CompiledModel,
    names: list[str],
    moment_fn: Callable[..., Any],
    equilibrium: tuple[str, Any, Callable[..., Any]] | None,
    theta: Array,
    n_types: int = 1,
) -> tuple[Array, Array]:
    """Model moments and exact Jacobian at ``theta`` (clearing the market first if asked)."""
    at = model.with_params(**dict(zip(names, theta, strict=True)))
    if n_types > 1:
        if equilibrium is not None:
            raise ValidationError("discount-factor types need partial equilibrium (no market)")
        return mixture_moments(at, names, moment_fn, n_types)
    if equilibrium is None:
        return stationary_moments(at, names, moment_fn)
    price, bracket, excess_fn = equilibrium

    def excess(r: float, sol: Any, mu: Array) -> tuple[float, dict[str, float]]:
        m = at.with_params(**{price: r})
        z = m.shock_process(sol.model.problem.shock).values
        args = (mu, sol.policy["consumption"], sol.policy["assets_next"], sol.grids["assets"], z)
        return float(excess_fn(*args, m.p)), {}

    eq = solve_stationary_equilibrium(
        lambda r: at.with_params(**{price: r}),
        price,
        bracket(at) if callable(bracket) else bracket,
        excess,
        tol=1e-10,
        solve_options={"tol": 1e-11, "max_iter": 100_000},
    )
    assert eq.model is not None
    m_eq, G, _ = equilibrium_moments(eq.model, names, price, excess_fn, moment_fn)
    return m_eq, G


def smm_exact(  # noqa: C901 - two-step estimation with inference; steps are inline
    model: CompiledModel,
    free_params: Sequence[str],
    data_moments: npt.ArrayLike,
    moment_fn: Callable[..., Any],
    omega: Array,
    *,
    n_obs: int,
    weights: str | Array = "optimal",
    equilibrium: tuple[str, Any, Callable[..., Any]] | None = None,
    max_iter: int = 200,
    starts: Sequence[npt.ArrayLike] = (),
    n_types: int = 1,
) -> SMMResult:
    """Estimate ``free_params`` so that stationary-distribution moments match ``data_moments``.

    ``moment_fn`` has the signature of :func:`stationary_moments`. ``omega`` is the asymptotic
    covariance of ``sqrt(n_obs) * (data moments)`` (from the data; there is no simulation noise).
    ``weights`` is ``"identity"``, ``"optimal"`` (two-step, ``W = omega^-1``) or a matrix. Pass
    ``equilibrium=(price, bracket, excess_fn)`` to clear a market at every parameter value
    (see :func:`equilibrium_moments`); ``bracket`` is a ``(lo, hi)`` pair or a function of the
    model (the upper end usually depends on the discount factor). Bounds come from the parameter
    set; the optimiser is L-BFGS-B with the exact gradient ``2 G' W g``. Parameter values where
    the model cannot be solved get a large penalty (counted in ``diagnostics["failed_evals"]``);
    tight bounds help. A failure at the starting point raises. ``starts`` are extra start vectors
    (ordered as ``free_params``); the first-step fit is run from each and the lowest loss is kept,
    which guards against a poor local minimum (unusable extra starts are skipped).
    ``n_types > 1`` estimates a population of discount-factor types (``beta`` +/-
    ``beta_spread``; see :func:`mixture_moments`).
    """
    names = list(free_params)
    m_hat = np.asarray(data_moments, dtype=np.float64).ravel()
    q, p = m_hat.size, len(names)
    omega = np.asarray(omega, dtype=np.float64)
    if q < p or omega.shape != (q, q):
        raise ValidationError(f"need q >= p moments and a ({q}, {q}) omega; got q={q}, p={p}")
    ps = model.parameters
    bounds = ps.bounds(names)
    counters = {"evals": 0, "failed": 0, "ok": 0}

    def evaluate(theta: Array) -> tuple[Array, Array]:
        counters["evals"] += 1
        return _evaluate(model, names, moment_fn, equilibrium, theta, n_types)

    def fit(W: Array, start: Array) -> Any:
        def fun(theta: Array) -> tuple[float, Array]:
            try:
                m, G = evaluate(theta)
                if not (np.isfinite(m).all() and np.isfinite(G).all()):
                    raise ConvergenceError("non-finite moments")
            except DyneconError:
                if not counters["ok"]:  # the starting point itself must be valid
                    raise
                counters["failed"] += 1
                return PENALTY, np.zeros_like(theta)  # the line search backs away from it
            counters["ok"] = 1
            g = m - m_hat
            return float(g @ W @ g), 2.0 * G.T @ W @ g

        return minimize(
            fun, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter}
        )

    wkind, W = _weights(weights, q)
    res = fit(W, ps.vector(names))
    for extra in starts:
        try:
            alt = fit(W, np.asarray(extra, dtype=np.float64))
        except DyneconError:
            continue
        res = alt if alt.fun < res.fun else res
    if weights == "optimal":
        W = np.asarray(np.linalg.pinv(omega), dtype=np.float64)
        res = fit(W, res.x)
    theta = np.asarray(res.x, dtype=np.float64)
    m_model, G = evaluate(theta)
    g = m_model - m_hat
    bread = np.linalg.pinv(G.T @ W @ G)
    cov = bread @ G.T @ W @ omega @ W @ G @ bread / n_obs
    df = q - p
    j_stat = float(n_obs * g @ np.linalg.pinv(omega) @ g) if df > 0 else None
    return SMMResult(
        names=names,
        estimates=theta,
        std_errors=np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        loss=float(g @ W @ g),
        weights=W,
        omega=omega,
        jacobian=G,
        model_moments=m_model,
        data_moments=m_hat,
        n_obs=n_obs,
        n_sim=float("inf"),
        j_stat=j_stat,
        j_pvalue=None if j_stat is None else float(stats.chi2.sf(j_stat, df)),
        df=df,
        converged=bool(res.success),
        diagnostics={
            "n_evals": counters["evals"],
            "failed_evals": counters["failed"],
            "optimizer_message": str(res.message),
        },
        config={"method": "L-BFGS-B (exact gradient)", "weights": wkind, "n_obs": n_obs},
        loss_fn=lambda params: float(
            (lambda d: d @ W @ d)(evaluate(np.array([params[n] for n in names]))[0] - m_hat)
        ),
    )


def quasi_posterior(  # noqa: C901 - one log-density with its chain rule; steps are inline
    model: CompiledModel,
    free_params: Sequence[str],
    data_moments: npt.ArrayLike,
    moment_fn: Callable[..., Any],
    omega: Array,
    *,
    n_obs: int,
    equilibrium: tuple[str, Any, Callable[..., Any]] | None = None,
    n_draws: int = 500,
    n_chains: int = 4,
    n_warmup: int = 300,
    rng: np.random.Generator | int | None = None,
    n_types: int = 1,
) -> MCMCResult:
    """Draw from the quasi-posterior ``prior * exp(-n/2 g' W g)`` with HMC (``W = omega^-1``).

    This is the Chernozhukov-Hong quasi-posterior of the moment conditions, not a likelihood.
    It uses the same exact moment Jacobians as :func:`smm_exact`, so every leapfrog step costs
    one model solve plus one derivative. Sampling is on each parameter's unconstrained scale
    (log-Jacobian included); parameters without a prior are flat on their bounds. Draws are
    returned in parameter space. Parameter values where the model cannot be solved have
    density zero. With ``W = omega^-1`` and a flat prior the posterior spread approximates the
    :func:`smm_exact` standard errors.
    """
    names = list(free_params)
    m_hat = np.asarray(data_moments, dtype=np.float64).ravel()
    omega = np.asarray(omega, dtype=np.float64)
    if m_hat.size < len(names) or omega.shape != (m_hat.size, m_hat.size):
        raise ValidationError("need q >= p moments and a (q, q) omega")
    W = np.asarray(np.linalg.pinv(omega), dtype=np.float64)
    ps = model.parameters
    p = len(names)

    def logp_grad(u: Array) -> tuple[float, Array]:
        with np.errstate(all="ignore"):  # wild leapfrog steps overflow the transforms: density 0
            theta = ps.from_unconstrained(u, names)
            lp, dprior = float(ps.log_jacobian(u, names)), np.zeros(p)
        for i, (n, v) in enumerate(zip(names, theta, strict=True)):
            prior = ps[n].prior
            if prior is not None:
                h = 1e-6 * max(1.0, abs(float(v)))
                lp += prior.logpdf(float(v))
                dprior[i] = (prior.logpdf(float(v) + h) - prior.logpdf(float(v) - h)) / (2 * h)
        if not np.isfinite(lp) or not np.all(np.isfinite(dprior)):
            return float("-inf"), np.zeros(p)
        try:
            m, G = _evaluate(model, names, moment_fn, equilibrium, theta, n_types)
        except DyneconError:
            return float("-inf"), np.zeros(p)
        g = m - m_hat
        if not (np.isfinite(g).all() and np.isfinite(G).all()):
            return float("-inf"), np.zeros(p)
        grad_theta = -n_obs * (G.T @ W @ g) + dprior
        h = 1e-6
        dtheta = (ps.from_unconstrained(u + h, names) - ps.from_unconstrained(u - h, names)) / (
            2 * h
        )
        eye = np.eye(p) * h  # the log-Jacobian is a sum, so differentiate it coordinate-wise
        dlogj = np.array(
            [(ps.log_jacobian(u + e, names) - ps.log_jacobian(u - e, names)) / (2 * h) for e in eye]
        )
        return lp - 0.5 * n_obs * float(g @ W @ g), grad_theta * dtheta + dlogj

    u0 = ps.to_unconstrained(ps.vector(names), names)
    res = hmc(logp_grad, u0, n_draws, n_chains=n_chains, n_warmup=n_warmup, rng=rng, names=names)
    theta_draws = np.empty_like(res.draws)
    for c in range(res.draws.shape[0]):
        for t in range(res.draws.shape[1]):
            theta_draws[c, t] = ps.from_unconstrained(res.draws[c, t], names)
    rhat = np.array([split_rhat(theta_draws[:, :, i]) for i in range(p)])
    ess_v = np.array([ess(theta_draws[:, :, i]) for i in range(p)])
    return MCMCResult(
        theta_draws, names, res.accept_rate, rhat, ess_v, res.log_posterior, res.divergences
    )
