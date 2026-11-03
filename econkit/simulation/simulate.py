"""Panel and time-series simulation of solved ``saving`` models.

Economics note:
Simulation turns a solved policy into a cross-section/time series of households. All
shock draws happen up front in NumPy, then a deterministic kernel evaluates the policy by
interpolation, so results do not depend on thread count or backend (up to rounding).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.core.solution import Solution
from econkit.model.problems import SavingProblem
from econkit.numerics._dispatch import resolve_backend
from econkit.numerics._registry import register
from econkit.numerics.markov import markov_paths, stationary_distribution
from econkit.simulation.kernels import simulate_saving_numba, simulate_saving_numpy
from econkit.simulation.result import SimulationResult
from econkit.utils.jit import NUMBA_AVAILABLE
from econkit.utils.rng import as_generator

__all__ = ["simulate"]

register("simulate_saving", "numpy", simulate_saving_numpy)
if NUMBA_AVAILABLE:
    register("simulate_saving", "numba", simulate_saving_numba)


def _saving_parts(solution: Solution) -> tuple[Any, SavingProblem]:
    model = solution.model
    prob = None if model is None else model.problem
    if model is None or not isinstance(prob, SavingProblem):
        raise ValidationError("simulate() needs a Solution of a 'saving' problem")
    return model, prob


def draw_shock_paths(
    P: npt.NDArray[np.float64],
    n_total: int,
    n_agents: int,
    gen: np.random.Generator,
    backend: str | None,
) -> npt.NDArray[np.int64]:
    """Shock-state indices ``(n_total, n_agents)``, first state drawn from the stationary law."""
    if P.shape[0] == 1:
        return np.zeros((n_total, n_agents), dtype=np.int64)
    cdf = np.cumsum(stationary_distribution(P))
    cdf[-1] = 1.0
    s0 = np.searchsorted(cdf, gen.random(n_agents), side="right").astype(np.int64)
    u = gen.random((n_total - 1, n_agents))
    return markov_paths(P, n_total, s0, u, backend=backend)


def simulate(
    solution: Solution,
    *,
    n_periods: int,
    n_agents: int = 1,
    burn_in: int = 0,
    init: float | npt.ArrayLike | None = None,
    rng: np.random.Generator | int | None = None,
    shocks: npt.NDArray[np.int64] | None = None,
    store: list[str] | None = None,
    backend: str | None = None,
) -> SimulationResult:
    """Simulate ``n_agents`` households for ``n_periods`` (after ``burn_in`` discarded periods).

    ``init`` is initial assets (scalar or per-agent; default: the lower bound of the asset
    grid). ``shocks`` optionally supplies shock-state indices ``(burn_in + n_periods, n_agents)``
    (common random numbers); otherwise they are drawn from ``rng``. ``store`` limits the
    returned variables (default all).
    """
    model, prob = _saving_parts(solution)
    if n_periods < 1 or n_agents < 1 or burn_in < 0:
        raise ValidationError("need n_periods >= 1, n_agents >= 1, burn_in >= 0")
    total = burn_in + n_periods
    gen = as_generator(rng)
    grid = np.ascontiguousarray(solution.grids[prob.asset])
    shock = model.shock_process(prob.shock)
    if shocks is None:
        idx = draw_shock_paths(shock.P, total, n_agents, gen, backend)
    else:
        idx = np.ascontiguousarray(shocks, dtype=np.int64)
        if idx.shape != (total, n_agents):
            raise ValidationError(f"shocks must have shape {(total, n_agents)}, got {idx.shape}")
        if np.min(idx) < 0 or np.max(idx) >= shock.values.shape[0]:
            raise ValidationError("shocks contain out-of-range state indices")
    a0 = (
        np.full(n_agents, grid[0])
        if init is None
        else np.broadcast_to(init, (n_agents,)).astype(np.float64)
    )
    pol_a = np.ascontiguousarray(solution.policy[f"{prob.asset}_next"], dtype=np.float64)
    pol_c = np.ascontiguousarray(solution.policy[prob.control], dtype=np.float64)
    backend_used, kernel = resolve_backend("simulate_saving", backend=backend)
    assets, cons = kernel(grid, pol_a, pol_c, idx, np.ascontiguousarray(a0))
    sl = slice(burn_in, None)
    paths: dict[str, npt.NDArray[Any]] = {
        prob.asset: assets[sl],
        prob.control: cons[sl],
        "shock_index": idx[sl],
    }
    if prob.shock is not None:
        paths[prob.shock] = shock.values[idx[sl]]
    if store is not None:
        missing = [s for s in store if s not in paths]
        if missing:
            raise ValidationError(f"cannot store unknown variables {missing}; have {sorted(paths)}")
        paths = {k: v for k, v in paths.items() if k in store}
    entropy = getattr(gen.bit_generator.seed_seq, "entropy", None)  # type: ignore[attr-defined]
    return SimulationResult(
        paths=paths,
        n_periods=n_periods,
        n_agents=n_agents,
        seed_entropy=int(entropy) if isinstance(entropy, int) else None,
        solution_fingerprint=str(solution.metadata.get("fingerprint", "")),
        metadata={"backend": backend_used, "burn_in": burn_in},
    )
