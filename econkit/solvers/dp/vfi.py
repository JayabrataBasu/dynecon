"""Value function iteration for ``saving`` and ``finite_dp`` problems.

Economics note:
VFI iterates the Bellman operator, a contraction when the discount factor is below
one. For ``saving`` problems the payoff of every (assets, shock, next-assets) triple is
tabulated once, so each iteration is an array operation (or a compiled kernel).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ModelSpecificationError, ValidationError
from econkit.core.solution import Solution
from econkit.model.compiled import CompiledModel
from econkit.model.problems import FiniteDPProblem, SavingProblem
from econkit.numerics._dispatch import resolve_backend
from econkit.numerics._registry import register
from econkit.numerics.fixedpoint import fixed_point
from econkit.solvers.diagnostics import bellman_residual, euler_errors, policy_monotone_in_assets
from econkit.solvers.dp.kernels import bellman_saving_numba, bellman_saving_numpy
from econkit.solvers.registry import SOLVERS, resolve_config
from econkit.utils.jit import NUMBA_AVAILABLE
from econkit.utils.provenance import collect_provenance

__all__ = ["VFIConfig", "build_payoff_tensor", "solve_vfi"]

Array = npt.NDArray[np.float64]

register("bellman_saving", "numpy", bellman_saving_numpy)
if NUMBA_AVAILABLE:
    register("bellman_saving", "numba", bellman_saving_numba)


@dataclass(frozen=True)
class VFIConfig:
    """Options of :func:`solve_vfi`; the single source of VFI option names and defaults."""

    tol: float = 1e-8
    max_iter: int = 2000
    damping: float = 1.0
    howard_steps: int = 10
    monotone: bool = True
    max_memory_gb: float = 2.0
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.tol > 0.0:
            raise ValidationError("tol must be strictly positive")
        if self.max_iter < 1:
            raise ValidationError("max_iter must be >= 1")
        if not 0.0 < self.damping <= 1.0:
            raise ValidationError("damping must lie in (0, 1]")
        if self.howard_steps < 0:
            raise ValidationError("howard_steps must be >= 0")


def build_payoff_tensor(model: CompiledModel, max_memory_gb: float = 2.0) -> Array:
    """Payoff tensor ``U[j, i, k] = u(resources(a_i, z_j) - a_k)``, ``-inf`` where infeasible.

    Economics note:
    Tabulating the period payoff once turns the Bellman maximisation into pure array
    arithmetic; the user's utility and budget functions never run inside the iteration.
    """
    prob = model.problem
    assert isinstance(prob, SavingProblem)
    a = model.grids[prob.asset]
    z = model.shock_process(prob.shock).values
    n_a, n_z = a.shape[0], z.shape[0]
    need_gb = 8.0 * n_a * n_a * n_z / 1e9
    if need_gb > max_memory_gb:
        raise ValidationError(
            f"grid too large for the VFI payoff tensor ({need_gb:.2f} GB > {max_memory_gb} GB); "
            "use method='egm'"
        )
    m = np.broadcast_to(np.asarray(prob.resources(a[:, None], z[None, :], model.p)), (n_a, n_z))
    c = m.T[:, :, None] - a[None, None, :]  # [j, i, k]
    feasible = c > 0.0
    if not feasible.any(axis=2).all():
        raise ModelSpecificationError(
            "some (assets, shock) states have no feasible consumption choice on the grid; "
            "raise the resources or lower the asset grid's lower bound"
        )
    u = np.asarray(prob.utility(np.where(feasible, c, 1.0), model.p), dtype=np.float64)
    return np.where(feasible, u, -np.inf)


def _howard(
    v: Array, pol: npt.NDArray[np.int64], u_pol: Array, P: Array, beta: float, steps: int
) -> Array:
    cols = np.arange(v.shape[1])[None, :]
    for _ in range(steps):
        ev = beta * (v @ P.T)
        v = u_pol + ev[pol, cols]
    return v


def _solve_saving(model: CompiledModel, cfg: VFIConfig, backend: str | None) -> Solution:
    prob = model.problem
    assert isinstance(prob, SavingProblem)
    t0 = perf_counter()
    a = model.grids[prob.asset]
    shock = model.shock_process(prob.shock)
    P = shock.P
    beta = model.parameters.values[prob.discount]
    U = build_payoff_tensor(model, cfg.max_memory_gb)
    n_z, n_a = U.shape[0], U.shape[1]
    backend_used, kernel = resolve_backend("bellman_saving", backend=backend)
    state: dict[str, Any] = {"pol": np.zeros((n_a, n_z), dtype=np.int64)}

    def operator(v: Array) -> Array:
        evt = np.ascontiguousarray(beta * (v @ P.T)).T.copy()  # [j, k]
        v_new, pol = kernel(U, evt, cfg.monotone)
        state["pol"] = pol
        if cfg.howard_steps > 0:
            u_pol = np.take_along_axis(U, pol.T[:, :, None], axis=2)[:, :, 0].T
            v_new = _howard(v_new, pol, u_pol, P, beta, cfg.howard_steps)
        out: Array = v_new
        return out

    def progress(it: int, x: Array, r: float) -> None:
        del x
        if it % 10 == 0:
            print(f"vfi iter {it:5d}  change {r:.3e}  {perf_counter() - t0:.2f}s")

    callbacks = [progress] if cfg.verbose else []
    res = fixed_point(
        operator,
        np.zeros((n_a, n_z)),
        tol=cfg.tol,
        max_iter=cfg.max_iter,
        damping=cfg.damping,
        callbacks=callbacks,
        name="vfi",
    )
    v = res.x
    # final policy and a clean Bellman residual from one exact (no-Howard) application
    evt = np.ascontiguousarray(beta * (v @ P.T)).T.copy()
    v_chk, pol = kernel(U, evt, cfg.monotone)
    a_next = a[pol]
    m = np.broadcast_to(
        np.asarray(prob.resources(a[:, None], shock.values[None, :], model.p)), pol.shape
    )
    cons = m - a_next
    diagnostics = {
        "bellman_residual": bellman_residual(v_chk, v),
        "monotone_policy": policy_monotone_in_assets(a_next),
        "max_feasibility_violation": float(max(0.0, -cons.min())),
        "n_binding": float(np.mean(pol == 0)),
        "residual_history": list(res.history),
    }
    sol = Solution(
        method="vfi",
        converged=res.converged,
        iterations=res.iterations,
        residual=res.residual,
        elapsed=perf_counter() - t0,
        backend=backend_used,
        model_name=model.name,
        params=model.parameters.values,
        grids={prob.asset: a, **({prob.shock: shock.values} if prob.shock else {})},
        policy={
            prob.control: cons,
            f"{prob.asset}_next": a_next,
            f"{prob.asset}_next_index": pol,
        },
        value=v,
        diagnostics=diagnostics,
        metadata={"fingerprint": model.fingerprint(), "provenance": collect_provenance()},
        model=model,
    )
    if prob.supports_egm:
        err = euler_errors(sol)
        if err.size:
            sol.diagnostics["euler_error_max"] = float(np.max(err))
            sol.diagnostics["euler_error_mean"] = float(np.mean(err))
    return sol


def _solve_finite_dp(model: CompiledModel, cfg: VFIConfig) -> Solution:
    prob = model.problem
    assert isinstance(prob, FiniteDPProblem)
    t0 = perf_counter()
    R = np.asarray(prob.reward(model.p), dtype=np.float64)  # [S, A]
    T = np.asarray(prob.transition(model.p), dtype=np.float64)  # [A, S, S']
    beta = model.parameters.values[prob.discount]

    def q_values(v: Array) -> Array:
        return np.asarray(R + beta * (T @ v).T, dtype=np.float64)

    res = fixed_point(
        lambda v: q_values(v).max(axis=1),
        np.zeros(R.shape[0]),
        tol=cfg.tol,
        max_iter=cfg.max_iter,
        damping=cfg.damping,
        name="vfi",
    )
    q = q_values(res.x)
    return Solution(
        method="vfi",
        converged=res.converged,
        iterations=res.iterations,
        residual=res.residual,
        elapsed=perf_counter() - t0,
        backend="numpy",
        model_name=model.name,
        params=model.parameters.values,
        grids={},
        policy={"action": q.argmax(axis=1).astype(np.int64)},
        value=res.x,
        diagnostics={
            "bellman_residual": bellman_residual(q.max(axis=1), res.x),
            "residual_history": list(res.history),
        },
        metadata={"fingerprint": model.fingerprint(), "provenance": collect_provenance()},
        model=model,
    )


def solve_vfi(model: CompiledModel, *, backend: str | None = None, **options: Any) -> Solution:
    """Value function iteration (see :class:`VFIConfig` for options)."""
    cfg = resolve_config(VFIConfig, options)
    out = (
        _solve_saving(model, cfg, backend)
        if isinstance(model.problem, SavingProblem)
        else _solve_finite_dp(model, cfg)
    )
    out.metadata["config"] = cfg
    return out


SOLVERS["vfi"] = solve_vfi
