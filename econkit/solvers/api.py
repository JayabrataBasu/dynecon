"""The single public entry point for solving models.

Economics note:
``solve`` hides whether a model is solved by value iteration or by the endogenous
grid method; the resolved method, backend and options are recorded on the result.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from econkit.core.errors import ValidationError
from econkit.core.solution import Solution
from econkit.model.compiled import CompiledModel
from econkit.model.problems import DSGEProblem, PricingProblem, SavingProblem
from econkit.solvers import dp, egm, perturbation, pricing  # noqa: F401  (register)
from econkit.solvers.registry import SOLVERS

__all__ = ["solve"]


def solve(
    model: CompiledModel, method: str = "auto", *, backend: str | None = None, **options: Any
) -> Solution:
    """Solve ``model`` with ``method`` (``"auto"``, ``"vfi"``, ...) and keyword ``options``.

    Options are validated against the method's frozen config dataclass, which is recorded
    in ``Solution.metadata["config"]``. Non-convergence warns (``ConvergenceWarning``) and
    sets ``converged=False``; with ``config.strict_convergence`` it raises instead.
    """
    if not isinstance(model, CompiledModel):
        raise ValidationError("solve() needs a CompiledModel; call Model.compile() first")
    chosen = _auto_method(model) if method == "auto" else method
    if chosen not in SOLVERS:
        raise ValidationError(f"unknown method {method!r}; available: {sorted(SOLVERS)}")
    sol = SOLVERS[chosen](model, backend=backend, **options)
    cfg = sol.metadata.get("config")
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        sol.metadata["config"] = dataclasses.asdict(cfg)
    sol.metadata["requested_method"] = method
    return sol


def _auto_method(model: CompiledModel) -> str:
    """EGM when the saving problem supplies the marginal fields, otherwise VFI."""
    prob = model.problem
    if isinstance(prob, DSGEProblem):
        return "perturbation"
    if isinstance(prob, PricingProblem):
        return "pricing"
    return "egm" if isinstance(prob, SavingProblem) and prob.supports_egm else "vfi"
