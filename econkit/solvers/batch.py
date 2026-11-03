"""Solve many parameterisations of one model in a single vectorised JAX call.

Economics note:
Comparative statics and estimation solve the same household problem at many parameter
values; batching them with ``vmap`` replaces a Python loop by one compiled computation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ConvergenceWarning, ValidationError
from econkit.core.solution import Solution
from econkit.model.compiled import CompiledModel
from econkit.model.problems import SavingProblem
from econkit.solvers.diagnostics import euler_errors, policy_monotone_in_assets
from econkit.utils.provenance import collect_provenance

__all__ = ["solve_many"]

Array = npt.NDArray[np.float64]


def _points(
    grid: Mapping[str, Sequence[float]] | Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    if isinstance(grid, Mapping):
        names = list(grid)
        sizes = {len(v) for v in grid.values()}
        if len(sizes) != 1:
            raise ValidationError("all parameter lists in `param_grid` must have the same length")
        return [dict(zip(names, vals, strict=True)) for vals in zip(*grid.values(), strict=True)]
    return [dict(p) for p in grid]


def _check_batchable(model: CompiledModel, others: list[CompiledModel]) -> None:
    """Parameters that change a shock process cannot be batched (the chain is fixed)."""
    for m in others:
        for s in model.shocks:
            same = np.array_equal(m.shocks[s].values, model.shocks[s].values) and np.array_equal(
                m.shocks[s].P, model.shocks[s].P
            )
            if not same:
                raise ValidationError("a varied parameter changes a shock process; solve singly")


def solve_many(
    model: CompiledModel,
    param_grid: Mapping[str, Sequence[float]] | Sequence[Mapping[str, float]],
    *,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> list[Solution]:
    """EGM-solve ``model`` at every parameter point with ``jax.vmap`` (one compiled loop).

    ``param_grid`` is a mapping of equal-length lists (zipped) or a list of dicts. Only
    parameters that enter the template callables can vary (not those feeding grids or shock
    processes); the callables must be JAX-traceable. Returns one :class:`Solution` per point.
    """
    import jax
    import jax.numpy as jnp

    from econkit import _jax_setup  # noqa: F401  (enables float64)
    from econkit.model.specs import ParamView
    from econkit.numerics.backends.jax_kernels import interp_linear

    prob = model.problem
    if not isinstance(prob, SavingProblem) or not prob.supports_egm:
        raise ValidationError("solve_many needs a 'saving' problem with marginal fields (EGM)")
    mu_fn, inv_fn, ret_fn = (
        prob.marginal_utility,
        prob.inverse_marginal_utility,
        prob.marginal_return,
    )
    assert mu_fn is not None and inv_fn is not None and ret_fn is not None
    points = _points(param_grid)
    if not points:
        raise ValidationError("param_grid is empty")
    names = sorted({k for p in points for k in p})
    models_b = [model.with_params(**p) for p in points]
    _check_batchable(model, models_b)
    base = model.parameters.values
    values = jnp.asarray([[p.get(n, base[n]) for n in names] for p in points])
    a = jnp.asarray(model.grids[prob.asset])
    shock = model.shock_process(prob.shock)
    z, P = jnp.asarray(shock.values), jnp.asarray(shock.P)
    n_a, n_z = a.shape[0], z.shape[0]
    a_min = a[0]

    def solve_one(vec: Any) -> tuple[Any, Any, Any]:
        pv = dict(base)
        pv.update({n: vec[i] for i, n in enumerate(names)})
        p = ParamView(pv)
        beta = pv[prob.discount]
        m_grid = jnp.broadcast_to(prob.resources(a[:, None], z[None, :], p), (n_a, n_z))
        gross = jnp.broadcast_to(ret_fn(a[:, None], z[None, :], p), (n_a, n_z))

        def operator(C: Any) -> Any:
            rhs = beta * ((gross * mu_fn(C, p)) @ P.T)
            c_end = inv_fn(rhs, p)
            m_end = c_end + a[:, None]

            def column(j: Any) -> Any:
                m_min = m_grid[0, j]
                constrained = m_min < m_end[0, j]
                x0 = jnp.where(constrained, m_min, m_end[0, j] - 1.0)
                f0 = jnp.where(constrained, m_min - a_min, c_end[0, j] - 1.0)
                xk = jnp.concatenate([x0[None], m_end[:, j]])
                fk = jnp.concatenate([f0[None], c_end[:, j]])
                return interp_linear(m_grid[:, j], xk, fk, 1)

            return jax.vmap(column)(jnp.arange(n_z)).T

        def cond(state: tuple[Any, Any, Any]) -> Any:
            _, res, it = state
            return (res >= tol) & (it < max_iter)

        def body(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            C, _, it = state
            C_new = operator(C)
            return C_new, jnp.max(jnp.abs(C_new - C)), it + 1

        C0 = jnp.maximum(m_grid - a_min, 1e-8)
        C, res, it = jax.lax.while_loop(cond, body, (C0, jnp.asarray(jnp.inf), jnp.asarray(0)))
        return C, res, it

    t0 = perf_counter()
    C_b, res_b, it_b = jax.jit(jax.vmap(solve_one))(values)
    elapsed = perf_counter() - t0
    out: list[Solution] = []
    for b, m in enumerate(models_b):
        C = np.asarray(C_b[b], dtype=np.float64)
        a_np = m.grids[prob.asset]
        m_grid = np.broadcast_to(
            np.asarray(
                prob.resources(a_np[:, None], m.shock_process(prob.shock).values[None, :], m.p)
            ),
            C.shape,
        )
        converged = bool(res_b[b] < tol)
        if not converged:
            warnings.warn(
                f"solve_many point {b} did not converge", ConvergenceWarning, stacklevel=2
            )
        sol = Solution(
            method="egm",
            converged=converged,
            iterations=int(it_b[b]),
            residual=float(res_b[b]),
            elapsed=elapsed / len(models_b),
            backend="jax",
            model_name=m.name,
            params=m.parameters.values,
            grids={prob.asset: a_np, **({prob.shock: shock.values} if prob.shock else {})},
            policy={prob.control: C, f"{prob.asset}_next": m_grid - C, "cash_on_hand": m_grid},
            value=None,
            diagnostics={"monotone_policy": policy_monotone_in_assets(m_grid - C)},
            metadata={
                "fingerprint": m.fingerprint(),
                "provenance": collect_provenance(),
                "config": {"tol": tol, "max_iter": max_iter},
                "requested_method": "solve_many",
            },
            model=m,
        )
        err = euler_errors(sol)
        if err.size:
            sol.diagnostics["euler_error_max"] = float(np.max(err))
            sol.diagnostics["euler_error_mean"] = float(np.mean(err))
        out.append(sol)
    return out
