"""Welfare: policy values and consumption-equivalent variation (CEV).

Economics note:
CEV answers "by what percentage would consumption have to rise in every period and state
under the baseline to make a household as well off as under the reform?" A positive CEV
means the reform is welfare improving; comparisons are between stationary equilibria.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from econkit.core.errors import ValidationError
from econkit.core.solution import Solution
from econkit.model.problems import SavingProblem
from econkit.numerics.interpolation import linear_interp_1d
from econkit.simulation.distribution import build_transition_matrix

__all__ = ["cev", "cev_from_values", "evaluate_policy_value"]

Array = npt.NDArray[np.float64]


def evaluate_policy_value(solution: Solution) -> Array:
    """Value of following the policy: ``V = (I - beta T)^{-1} u`` on the (asset, shock) grid.

    Works for any ``saving`` solution, including EGM solutions that carry no value function.
    """
    model = solution.model
    prob = None if model is None else model.problem
    if model is None or not isinstance(prob, SavingProblem):
        raise ValidationError("evaluate_policy_value needs a Solution of a 'saving' problem")
    cons = solution.policy[prob.control]
    if np.any(cons <= 0.0):
        raise ValidationError("policy has non-positive consumption")
    beta = model.parameters.values[prob.discount]
    u = np.asarray(prob.utility(cons, model.p), dtype=np.float64)
    T = build_transition_matrix(solution)
    n = T.shape[0]
    v = spsolve((identity(n, format="csc") - beta * T).tocsc(), u.ravel())
    return np.asarray(v, dtype=np.float64).reshape(cons.shape)


def cev_from_values(
    v_base: npt.ArrayLike, v_reform: npt.ArrayLike, sigma: float, beta: float
) -> Array:
    """CEV ``lambda`` with ``V_base(c (1 + lambda)) = V_reform`` for CRRA preferences.

    ``sigma != 1``: ``lambda = (V_reform / V_base)^(1/(1-sigma)) - 1`` (values of the same
    sign); ``sigma == 1``: ``lambda = exp((1 - beta)(V_reform - V_base)) - 1``.
    """
    vb, vr = np.asarray(v_base, dtype=np.float64), np.asarray(v_reform, dtype=np.float64)
    if sigma == 1.0:
        return np.asarray(np.expm1((1.0 - beta) * (vr - vb)), dtype=np.float64)
    if np.any(vb * vr <= 0.0):
        raise ValidationError("CRRA CEV needs base and reform values of the same sign")
    return np.asarray((vr / vb) ** (1.0 / (1.0 - sigma)) - 1.0, dtype=np.float64)


def cev(base: Solution, reform: Solution, distribution: Array | None = None) -> Array | float:
    """CEV of a reform relative to a baseline solution (CRRA models with a ``sigma`` parameter).

    With ``distribution`` (the baseline stationary mass on the baseline grid) the reform value
    function is interpolated onto the baseline grid and the aggregate CEV is computed from
    expected values over that cross-section: the welfare gain for the households alive when
    the reform is introduced, ignoring the transition. Without it a state-wise array is
    returned (both solutions must share grids).

    Economics note:
    Comparing average values over each regime's *own* stationary distribution would mix the
    policy's effect with the change in who holds how much wealth; evaluating both value
    functions on the baseline cross-section isolates the policy.
    """
    if base.model is None or reform.model is None:
        raise ValidationError("cev needs solutions that carry their models")
    if "sigma" not in base.model.parameters:
        raise ValidationError("cev needs a CRRA model with a parameter named 'sigma'")
    sigma = base.model.parameters.values["sigma"]
    prob = base.model.problem
    assert isinstance(prob, SavingProblem)
    beta = base.model.parameters.values[prob.discount]
    vb = base.value if base.value is not None else evaluate_policy_value(base)
    vr = reform.value if reform.value is not None else evaluate_policy_value(reform)
    a_base, a_reform = base.grids[prob.asset], reform.grids[prob.asset]
    if distribution is not None:
        if vb.shape[1] != vr.shape[1]:
            raise ValidationError("baseline and reform must have the same shock states")
        vr_on_base = np.column_stack(
            [
                linear_interp_1d(a_base, a_reform, vr[:, j], extrapolate="linear")
                for j in range(vr.shape[1])
            ]
        )
        eb, er = float(np.sum(distribution * vb)), float(np.sum(distribution * vr_on_base))
        return float(cev_from_values(eb, er, sigma, beta))
    if vb.shape != vr.shape:
        raise ValidationError("state-wise CEV needs identical grids; pass the distribution")
    return cev_from_values(vb, vr, sigma, beta)
