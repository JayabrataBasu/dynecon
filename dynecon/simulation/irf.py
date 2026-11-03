"""Generalised impulse responses of solved nonlinear models.

Economics note:
A generalised IRF is the average difference between a path that starts from a shocked
state and one that does not, using common random numbers afterwards. It measures how an
aggregate responds to a shock at typical points of the ergodic distribution.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.core.solution import Solution
from dynecon.numerics.markov import markov_paths
from dynecon.simulation.simulate import _saving_parts, simulate
from dynecon.utils.rng import as_generator

__all__ = ["simulate_irf"]


def simulate_irf(
    solution: Solution,
    *,
    shock_steps: int = 1,
    horizon: int = 40,
    n_paths: int = 2000,
    burn_in: int = 200,
    log: bool = False,
    rng: np.random.Generator | int | None = None,
    backend: str | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Mean response of each simulated variable to moving the shock state by ``shock_steps``.

    Paths start from the ergodic distribution (after ``burn_in``); the shocked path starts
    with the shock index moved by ``shock_steps`` grid points; afterwards both use identical
    uniform draws. Entry ``t`` is the mean difference (shocked minus baseline) at horizon ``t``
    (of logs if ``log=True``). Paths whose shifted shock index leaves the grid are dropped.
    """
    model, prob = _saving_parts(solution)
    if prob.shock is None:
        raise ValidationError("the model has no shock to perturb")
    if horizon < 1 or n_paths < 1:
        raise ValidationError("need horizon >= 1 and n_paths >= 1")
    gen = as_generator(rng)
    P = model.shock_process(prob.shock).P
    n_z = P.shape[0]
    start = simulate(
        solution, n_periods=1, n_agents=n_paths, burn_in=burn_in, rng=gen, backend=backend
    )
    a0 = start.paths[prob.asset][0]
    j0 = start.paths["shock_index"][0]
    j1 = j0 + shock_steps
    ok = (j1 >= 0) & (j1 < n_z)
    if not ok.any():
        raise ValidationError("shock_steps moves every path off the shock grid")
    a0, j0, j1 = a0[ok], j0[ok], j1[ok]
    u = gen.random((horizon, a0.shape[0]))
    runs = []
    for j_init in (j0, j1):
        idx = markov_paths(P, horizon + 1, j_init.astype(np.int64), u, backend=backend)
        runs.append(
            simulate(
                solution,
                n_periods=horizon + 1,
                n_agents=a0.shape[0],
                init=a0,
                shocks=idx,
                backend=backend,
            )
        )
    base, shocked = runs
    out: dict[str, npt.NDArray[np.float64]] = {}
    for name in (prob.asset, prob.control, prob.shock):
        xb = base.paths[name].astype(np.float64)
        xs = shocked.paths[name].astype(np.float64)
        diff = np.log(xs) - np.log(xb) if log else xs - xb
        out[name] = diff.mean(axis=1)
    return out
