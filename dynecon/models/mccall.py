"""McCall job-search model (finite-state dynamic programming).

Economics note:
An unemployed worker draws a wage offer each period and either accepts (employed at that
wage forever) or rejects and collects the benefit ``c``. The optimal rule is a reservation
wage: accept iff the offer exceeds it. Higher benefits or patience raise the reservation
wage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dynecon.core.errors import ValidationError
from dynecon.core.solution import Solution
from dynecon.model import Model, Param
from dynecon.models.registry import register, split_overrides

__all__ = [
    "ANALYTIC",
    "CALIBRATION_SOURCE",
    "DEFAULTS",
    "PARAMS",
    "build",
    "reservation_wage",
    "reservation_wage_uniform",
]

CALIBRATION_SOURCE = "McCall (1970); textbook calibration with uniform offers on [0, 1]."

PARAMS: dict[str, Param] = {
    "beta": Param(0.96, bounds=(0.5, 0.9999), description="Discount factor"),
    "c": Param(0.5, bounds=(0.0, 2.0), description="Unemployment benefit"),
}
DEFAULTS: dict[str, Any] = {"n_w": 60}


def reservation_wage_uniform(beta: float, c: float) -> float:
    """Closed-form reservation wage for offers uniform on [0, 1].

    Solves ``k w^2 - (2k + 1) w + (k + c) = 0`` with ``k = beta / (2 (1 - beta))``.
    """
    k = beta / (2.0 * (1.0 - beta))
    return float(((2 * k + 1) - math.sqrt((2 * k + 1) ** 2 - 4 * k * (k + c))) / (2 * k))


ANALYTIC = {"reservation_wage_uniform": reservation_wage_uniform}


def _wages(n_w: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_w)


def build(**overrides: Any) -> Model:
    """McCall model as a ``finite_dp`` problem.

    States ``0..n_w-1``: unemployed holding offer ``w_i``; ``n_w..2 n_w-1``: employed at ``w_i``
    (absorbing). Actions: 0 = reject, 1 = accept.
    """
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    n = int(opts["n_w"])
    if n < 2:
        raise ValidationError("n_w must be >= 2")
    w = _wages(n)

    def reward(p: Any) -> np.ndarray:
        r = np.empty((2 * n, 2))
        r[:n, 0] = p.c
        r[:n, 1] = w
        r[n:, :] = w[:, None]
        return r

    def transition(p: Any) -> np.ndarray:
        t = np.zeros((2, 2 * n, 2 * n))
        t[0, :n, :n] = 1.0 / n
        t[1, np.arange(n), n + np.arange(n)] = 1.0
        t[:, n + np.arange(n), n + np.arange(n)] = 1.0
        return t

    m = Model("mccall")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    m.set_problem(
        "finite_dp",
        reward=reward,
        transition=transition,
        discount="beta",
        state_labels=tuple([f"offer_{i}" for i in range(n)] + [f"employed_{i}" for i in range(n)]),
        action_labels=("reject", "accept"),
    )
    return m


def reservation_wage(solution: Solution) -> float:
    """Lowest accepted wage in a McCall solution (``nan`` if every offer is rejected)."""
    n = solution.policy["action"].shape[0] // 2
    accept = solution.policy["action"][:n] == 1
    w = _wages(n)
    return float(w[accept].min()) if accept.any() else float("nan")


register("mccall", build)
