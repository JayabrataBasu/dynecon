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
import numpy as np
import numpy.typing as npt

from dynecon import _jax_setup  # noqa: F401  (enables float64)
from dynecon.core.errors import ConvergenceError, ValidationError
from dynecon.model.compiled import CompiledModel
from dynecon.model.problems import SavingProblem
from dynecon.model.specs import AR1Shock, MarkovShock, ParamView
from dynecon.simulation.distribution import lottery_weights, stationary_distribution_young
from dynecon.solvers.api import solve

__all__ = ["stationary_moments"]

Array = npt.NDArray[np.float64]
# shortcut: dense implicit solves, memory ~ states^2; upgrade to matrix-free GMRES for larger grids.
MAX_STATES = 4000


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
    return (jnp.exp(values) if shock.log else values), P


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
    operations (the built-in utilities are). Raises ``ConvergenceError`` if the JAX residuals do
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

    dC = -jnp.linalg.solve(jax.jacfwd(r1, 0)(C0, th0), jax.jacfwd(r1, 1)(C0, th0))
    dmu = -jnp.linalg.solve(
        jax.jacfwd(r2, 0)(mu0, C0, th0),
        jax.jacfwd(r2, 1)(mu0, C0, th0) @ dC + jax.jacfwd(r2, 2)(mu0, C0, th0),
    )
    d_mu, d_C, d_th = jax.jacfwd(moments, (0, 1, 2))(mu0, C0, th0)
    G = d_mu @ dmu + d_C @ dC + d_th
    return np.asarray(moments(mu0, C0, th0)), np.asarray(G)
