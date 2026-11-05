"""Exact derivatives of stationary heterogeneous-agent moments (implicit function theorem).

Economics note:
Moment-based estimation needs ``d moments / d parameters``. Finite differences of a solved
model are noisy and cost one solve per parameter; here the household fixed point and the
stationary distribution are differentiated exactly. The forward solve is Dynecon's own
NumPy/Numba EGM solver; JAX is used only to differentiate the residual equations
``C = T(C, theta)`` and ``mu = Lambda(C, theta)' mu``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import numpy as np
import numpy.typing as npt

from dynecon import _jax_setup  # noqa: F401  (enables float64)
from dynecon.core.errors import ConvergenceError, ValidationError
from dynecon.model.compiled import CompiledModel
from dynecon.model.problems import SavingProblem
from dynecon.model.specs import AR1Shock, MarkovShock, ParamView
from dynecon.simulation.distribution import lottery_weights, stationary_distribution_young
from dynecon.solvers.api import solve

__all__ = ["equilibrium_moments", "mixture_moments", "stationary_moments"]

Array = npt.NDArray[np.float64]
DENSE_STATES = 4000  # up to here the implicit systems are solved densely (exact, memory ~ n^2)
MAX_STATES = 200_000  # sanity cap; the GMRES path was checked against dense and at 6000 states


def _rouwenhorst(rho: Any, sigma_e: Any, n: int) -> tuple[Any, Any]:
    """JAX copy of :func:`dynecon.numerics.rouwenhorst` (differentiable in ``rho``, ``sigma_e``)."""
    p = (1.0 + rho) / 2.0
    trans = jnp.array([[p, 1.0 - p], [1.0 - p, p]])
    for m in range(3, n + 1):
        new = jnp.zeros((m, m))
        new = new.at[:-1, :-1].add(p * trans).at[:-1, 1:].add((1.0 - p) * trans)
        trans = new.at[1:, :-1].add((1.0 - p) * trans).at[1:, 1:].add(p * trans)
        trans = trans.at[1:-1, :].divide(2.0)
    psi = sigma_e / jnp.sqrt(1.0 - rho * rho) * jnp.sqrt(n - 1.0)
    return jnp.linspace(-psi, psi, n), trans / trans.sum(axis=1, keepdims=True)


def _shock(model: CompiledModel, name: str, vals: dict[str, Any]) -> tuple[Any, Any]:
    """Income values and transition matrix as functions of the parameter values ``vals``."""
    shock = next(s.shock for s in model.spec.states if s.name == name)
    if isinstance(shock, MarkovShock):
        return jnp.asarray(shock.values, dtype=float), jnp.asarray(shock.P, dtype=float)
    if not isinstance(shock, AR1Shock) or shock.method != "rouwenhorst":
        raise ValidationError(
            "exact derivatives support AR1Shock(method='rouwenhorst') and MarkovShock"
        )

    def get(x: float | str) -> Any:
        return vals[x] if isinstance(x, str) else x

    grid, P = _rouwenhorst(get(shock.rho), get(shock.sigma_e), shock.n_states)
    values = grid + shock.mean
    values = jnp.exp(values) if shock.log else values
    if shock.normalize:  # stationary distribution: solve (I - P') pi = 0 with sum(pi) = 1
        n = shock.n_states
        pi = jnp.linalg.solve((jnp.eye(n) - P.T).at[-1].set(1.0), jnp.zeros(n).at[-1].set(1.0))
        values = values / (pi @ values)
    return values, P


def _gmres(matvec: Callable[[Any], Any], rhs: Any) -> Any:
    """Solve ``A x = rhs`` for each column of ``rhs`` with GMRES given only ``v -> A v``."""
    cols = []
    for b in rhs.T:
        x, _ = jax.scipy.sparse.linalg.gmres(matvec, b, tol=1e-13, restart=80, maxiter=60)
        if float(jnp.abs(matvec(x) - b).max()) > 1e-9 * max(1.0, float(jnp.abs(b).max())):
            raise ConvergenceError("GMRES did not solve the implicit-derivative system")
        cols.append(x)
    return jnp.stack(cols, axis=1)


def _interp_columns(xq: Any, xk: Any, fk: Any) -> Any:
    """Column-wise linear interpolation with linear extrapolation (as ``linear_interp_columns``)."""

    def one(q: Any, x: Any, f: Any) -> Any:
        i = jnp.clip(jnp.searchsorted(x, q, side="right") - 1, 0, x.shape[0] - 2)
        return f[i] + (q - x[i]) / (x[i + 1] - x[i]) * (f[i + 1] - f[i])

    return jax.vmap(one, in_axes=(1, 1, 1), out_axes=1)(xq, xk, fk)


def stationary_moments(
    model: CompiledModel,
    free_params: Sequence[str],
    moment_fn: Callable[..., Any],
    *,
    tol: float = 1e-13,
    max_iter: int = 100_000,
) -> tuple[Array, Array]:
    """Moments of the stationary distribution and their exact Jacobian w.r.t. ``free_params``.

    ``moment_fn(mu, c, a_next, a_grid, z, p)`` is written with ``jax.numpy``; ``mu``, ``c`` and
    ``a_next`` have shape ``(n_assets, n_shocks)``, ``z`` holds the income values and ``p`` is the
    parameter view. Returns ``(m, G)`` with ``G[i, j] = d m_i / d free_params[j]`` (natural scale).

    The model must be a ``saving`` problem solvable by EGM whose equations use JAX-compatible
    operations (the built-in utilities are). Derivatives are those of ``model.with_params``: the
    grid and any ``MarkovShock`` are held fixed, and parameters act through the parameter view
    ``p`` and ``AR1Shock(rho, sigma_e)``. Raises ``ConvergenceError`` if the JAX residuals do
    not vanish at the NumPy solution, which also catches non-traceable model functions.
    """
    prob = model.problem
    if not isinstance(prob, SavingProblem) or not prob.supports_egm:
        raise ValidationError("exact derivatives need a 'saving' problem that supports EGM")
    shock_name, mu_fn = prob.shock, prob.marginal_utility
    inv_mu_fn, ret_fn = prob.inverse_marginal_utility, prob.marginal_return
    if shock_name is None or mu_fn is None or inv_mu_fn is None or ret_fn is None:
        raise ValidationError("exact derivatives need an income shock and EGM equations")
    names = list(free_params)
    base = dict(model.parameters.values)
    unknown = [n for n in names if n not in base]
    if unknown or not names:
        raise ValidationError(f"free_params must name model parameters, got {names}")
    a = jnp.asarray(model.grids[prob.asset])
    n_a = a.shape[0]
    n_z = model.shock_process(shock_name).values.shape[0]
    shape = (n_a, n_z)
    if n_a * n_z > MAX_STATES:
        raise ValidationError(
            f"exact derivatives support at most {MAX_STATES} states (got {n_a * n_z})"
        )

    sol = solve(model, method="egm", tol=tol, max_iter=max_iter)
    if not sol.converged:
        raise ConvergenceError("household problem did not converge; cannot differentiate")
    mu_np = stationary_distribution_young(sol, tol=tol)
    k = jnp.asarray(lottery_weights(sol)[0])
    cols = jnp.broadcast_to(jnp.arange(n_z)[None, :], shape)

    def primitives(th: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        vals = {**base, **{n: th[i] for i, n in enumerate(names)}}
        p = ParamView(vals)
        z, P = _shock(model, shock_name, vals)
        m_grid = jnp.broadcast_to(prob.resources(a[:, None], z[None, :], p), shape)
        gross = jnp.broadcast_to(ret_fn(a[:, None], z[None, :], p), shape)
        return p, vals[prob.discount], z, P, m_grid, gross

    def egm_residual(C: Any, th: Any) -> Any:
        p, beta, _, P, m_grid, gross = primitives(th)
        c_end = inv_mu_fn(beta * ((gross * mu_fn(C, p)) @ P.T), p)
        m_end = c_end + a[:, None]
        constrained = m_grid[0] < m_end[0]
        x0 = jnp.where(constrained, m_grid[0], m_end[0] - 1.0)
        f0 = jnp.where(constrained, m_grid[0] - a[0], c_end[0] - 1.0)
        xk, fk = jnp.vstack([x0[None], m_end]), jnp.vstack([f0[None], c_end])
        return (C - _interp_columns(m_grid, xk, fk)).ravel()

    def dist_residual(mu: Any, C: Any, th: Any) -> Any:
        _, _, _, P, m_grid, _ = primitives(th)
        a_c = jnp.clip(m_grid - C, a[0], a[-1])
        w = jnp.clip((a[k + 1] - a_c) / (a[k + 1] - a[k]), 0.0, 1.0)
        mass = mu.reshape(shape)
        moved = jnp.zeros(shape).at[k, cols].add(w * mass).at[k + 1, cols].add((1.0 - w) * mass)
        r = mu - (moved @ P).ravel()
        return r.at[-1].set(mu.sum() - 1.0)  # replace one redundant equation by sum(mu) = 1

    def moments(mu: Any, C: Any, th: Any) -> Any:
        p, _, z, _, m_grid, _ = primitives(th)
        c = C.reshape(shape)
        return jnp.ravel(jnp.asarray(moment_fn(mu.reshape(shape), c, m_grid - c, a, z, p)))

    th0 = jnp.asarray([base[n] for n in names])
    C0 = jnp.asarray(sol.policy[prob.control]).ravel()
    mu0 = jnp.asarray(mu_np).ravel()
    r1 = lambda C, th: egm_residual(C.reshape(shape), th)  # noqa: E731
    r2 = lambda mu, C, th: dist_residual(mu, C.reshape(shape), th)  # noqa: E731
    if max(float(jnp.abs(r1(C0, th0)).max()), float(jnp.abs(r2(mu0, C0, th0)).max())) > 1e-8:
        raise ConvergenceError(
            "JAX residuals do not vanish at the NumPy solution; model not traceable?"
        )

    if n_a * n_z <= DENSE_STATES:
        dC = -jnp.linalg.solve(jax.jacfwd(r1, 0)(C0, th0), jax.jacfwd(r1, 1)(C0, th0))
        dmu = -jnp.linalg.solve(
            jax.jacfwd(r2, 0)(mu0, C0, th0),
            jax.jacfwd(r2, 1)(mu0, C0, th0) @ dC + jax.jacfwd(r2, 2)(mu0, C0, th0),
        )
    else:  # matrix-free: only Jacobian-vector products, memory ~ states
        _, mv1 = jax.linearize(lambda C: r1(C, th0), C0)
        dC = -_gmres(mv1, jax.jacfwd(r1, 1)(C0, th0))
        _, mv2 = jax.linearize(lambda mu: r2(mu, C0, th0), mu0)
        by_C = jnp.stack([jax.jvp(lambda C: r2(mu0, C, th0), (C0,), (v,))[1] for v in dC.T], 1)
        dmu = -_gmres(mv2, by_C + jax.jacfwd(r2, 2)(mu0, C0, th0))
    d_mu, d_C, d_th = jax.jacfwd(moments, (0, 1, 2))(mu0, C0, th0)
    G = d_mu @ dmu + d_C @ dC + d_th
    return np.asarray(moments(mu0, C0, th0)), np.asarray(G)


def mixture_moments(
    model: CompiledModel,
    free_params: Sequence[str],
    moment_fn: Callable[..., Any],
    n_types: int,
    **options: Any,
) -> tuple[Array, Array]:
    """Moments of a population of ``n_types`` equally sized discount-factor types, with ``G``.

    Type ``k`` has ``beta + (k - (n_types - 1) / 2) * beta_spread`` (parameters ``beta`` and
    ``beta_spread`` of the model; the spread is typically small and non-negative). ``moment_fn``
    sees the pooled ``mu`` and the mass-weighted ``c`` and ``a_next`` of the whole population, so
    wealth
    shares and Gini coefficients are those of the mixture. Each type's exact Jacobian is combined by
    the chain rule, so the cost is ``n_types`` calls of :func:`stationary_moments`.
    """
    names = list(free_params)
    if n_types < 2 or "beta_spread" not in model.parameters.values:
        raise ValidationError("need n_types >= 2 and a model with a 'beta_spread' parameter")
    inner = [n for n in names if n != "beta_spread"]
    if "beta" not in inner:
        inner.append("beta")
    prob = model.problem
    assert isinstance(prob, SavingProblem) and prob.shock is not None
    shape = (model.grids[prob.asset].shape[0], model.shock_process(prob.shock).values.shape[0])
    size = shape[0] * shape[1]
    beta, spread = model.p.beta, model.p.beta_spread
    offsets = np.arange(n_types) - (n_types - 1) / 2.0
    shares = 1.0 / n_types
    stats, jacs = [], []

    def parts(mu: Any, c: Any, a_next: Any, _a: Any, _z: Any, _p: Any) -> Any:
        return jnp.concatenate([jnp.ravel(mu), jnp.ravel(mu * c), jnp.ravel(mu * a_next)])

    for off in offsets:
        sub = model.with_params(beta=float(beta + off * spread))
        m_k, J_k = stationary_moments(sub, inner, parts, **options)
        col = {n: J_k[:, i] for i, n in enumerate(inner)}
        cols = [off * col["beta"] if n == "beta_spread" else col[n] for n in names]
        stats.append(m_k)
        jacs.append(np.stack(cols, axis=1))

    a = jnp.asarray(model.grids[prob.asset])
    z = jnp.asarray(model.shock_process(prob.shock).values)

    def pooled(all_stats: Any) -> Any:
        tot = shares * jnp.sum(all_stats, axis=0)
        mu, mc, ma = tot[:size], tot[size : 2 * size], tot[2 * size :]
        safe = jnp.where(mu > 1e-300, mu, 1.0)
        c, a_next = jnp.where(mu > 1e-300, mc / safe, 0.0), jnp.where(mu > 1e-300, ma / safe, 0.0)
        return jnp.ravel(
            jnp.asarray(
                moment_fn(mu.reshape(shape), c.reshape(shape), a_next.reshape(shape), a, z, model.p)
            )
        )

    S = jnp.asarray(np.stack(stats))
    dM = jax.jacfwd(pooled)(S)  # (q, n_types, 3 * size)
    G = sum(np.asarray(dM[:, k, :]) @ jacs[k] for k in range(n_types))
    return np.asarray(pooled(S)), np.asarray(G)


def equilibrium_moments(
    model: CompiledModel,
    free_params: Sequence[str],
    price: str,
    excess_fn: Callable[..., Any],
    moment_fn: Callable[..., Any],
    **options: Any,
) -> tuple[Array, Array, Array]:
    """Moments at a stationary equilibrium and their exact total Jacobian, with ``dprice``.

    ``model`` is solved at the clearing price (parameter ``price``); ``excess_fn`` (same signature
    as ``moment_fn``) returns the scalar excess demand. By the implicit function theorem
    ``dprice/dtheta = -F_theta / F_price`` and ``dm/dtheta = m_theta + m_price * dprice/dtheta``,
    all taken from one :func:`stationary_moments` call. Returns ``(m, G, dprice)``. The price must
    enter the household equations through ``p`` (a wage computed from ``p.price`` is fine; one
    baked in when the model was built, as in the built-in ``aiyagari``, is not).
    """
    names = list(free_params)

    def both(*args: Any) -> Any:
        return jnp.concatenate(
            [jnp.ravel(jnp.asarray(excess_fn(*args))), jnp.ravel(jnp.asarray(moment_fn(*args)))]
        )

    m_all, G_all = stationary_moments(model, [price, *names], both, **options)
    if abs(m_all[0]) > 1e-6:
        raise ConvergenceError(f"model is not at equilibrium: excess demand {m_all[0]:.3g}")
    dprice = -G_all[0, 1:] / G_all[0, 0]
    return m_all[1:], G_all[1:, 1:] + np.outer(G_all[1:, 0], dprice), dprice
