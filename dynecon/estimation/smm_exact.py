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

from dynecon.core.errors import ValidationError
from dynecon.estimation.results import SMMResult
from dynecon.estimation.sensitivity import equilibrium_moments, stationary_moments
from dynecon.model.compiled import CompiledModel
from dynecon.solvers.equilibrium import solve_stationary_equilibrium

__all__ = ["smm_exact"]

Array = npt.NDArray[np.float64]


def smm_exact(
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
) -> SMMResult:
    """Estimate ``free_params`` so that stationary-distribution moments match ``data_moments``.

    ``moment_fn`` has the signature of :func:`stationary_moments`. ``omega`` is the asymptotic
    covariance of ``sqrt(n_obs) * (data moments)`` (from the data; there is no simulation noise).
    ``weights`` is ``"identity"``, ``"optimal"`` (two-step, ``W = omega^-1``) or a matrix. Pass
    ``equilibrium=(price, bracket, excess_fn)`` to clear a market at every parameter value
    (see :func:`equilibrium_moments`); ``bracket`` is a ``(lo, hi)`` pair or a function of the
    model (the upper end usually depends on the discount factor). Bounds come from the parameter
    set; the optimiser is L-BFGS-B with the exact gradient ``2 G' W g``. Keep the bounds tight
    enough that the household problem and the market bracket stay valid (a failed solve raises).
    """
    names = list(free_params)
    m_hat = np.asarray(data_moments, dtype=np.float64).ravel()
    q, p = m_hat.size, len(names)
    omega = np.asarray(omega, dtype=np.float64)
    if q < p or omega.shape != (q, q):
        raise ValidationError(f"need q >= p moments and a ({q}, {q}) omega; got q={q}, p={p}")
    ps = model.parameters
    bounds = ps.bounds(names)
    counters = {"evals": 0}

    def evaluate(theta: Array) -> tuple[Array, Array]:
        counters["evals"] += 1
        at = model.with_params(**dict(zip(names, theta, strict=True)))
        if equilibrium is None:
            return stationary_moments(at, names, moment_fn)
        price, bracket, excess_fn = equilibrium

        def excess(r: float, sol: Any, mu: Array) -> tuple[float, dict[str, float]]:
            m = at.with_params(**{price: r})
            z = m.shock_process(sol.model.problem.shock).values
            args = (
                mu,
                sol.policy["consumption"],
                sol.policy["assets_next"],
                sol.grids["assets"],
                z,
            )
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

    def fit(W: Array, start: Array) -> Any:
        def fun(theta: Array) -> tuple[float, Array]:
            m, G = evaluate(theta)
            g = m - m_hat
            return float(g @ W @ g), 2.0 * G.T @ W @ g

        return minimize(
            fun, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iter}
        )

    if isinstance(weights, str):
        if weights not in ("identity", "optimal"):
            raise ValidationError("weights must be 'identity', 'optimal' or a matrix")
        wkind, W = weights, np.eye(q)
    else:
        wkind, W = "custom", np.asarray(weights, dtype=np.float64)
    res = fit(W, ps.vector(names))
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
        diagnostics={"n_evals": counters["evals"], "optimizer_message": str(res.message)},
        config={"method": "L-BFGS-B (exact gradient)", "weights": wkind, "n_obs": n_obs},
        loss_fn=lambda params: float(
            (lambda d: d @ W @ d)(evaluate(np.array([params[n] for n in names]))[0] - m_hat)
        ),
    )
