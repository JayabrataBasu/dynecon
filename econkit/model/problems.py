"""Problem templates: the contract between a model and a family of solvers.

Economics note:
A template names exactly the ingredients a solver needs (preferences, budget
constraint, returns), so nothing in a solver ever calls an arbitrary equation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["FiniteDPProblem", "SavingProblem"]

Fn = Callable[..., Any]


@dataclass(frozen=True)
class SavingProblem:
    """One continuous asset, discrete/discretised shocks, one control.

    Callables are plain NumPy functions that broadcast over arrays::

        utility(c, p)  marginal_utility(c, p)  inverse_marginal_utility(x, p)
        resources(a, z, p)  marginal_return(a, z, p)

    ``resources`` is cash on hand as a function of *end-of-period* assets ``a`` and the
    shock ``z``; ``discount`` names the discount-factor parameter.
    """

    asset: str
    control: str
    discount: str
    utility: Fn
    resources: Fn
    shock: str | None = None
    marginal_utility: Fn | None = None
    inverse_marginal_utility: Fn | None = None
    marginal_return: Fn | None = None

    @property
    def supports_egm(self) -> bool:
        return None not in (
            self.marginal_utility,
            self.inverse_marginal_utility,
            self.marginal_return,
        )


@dataclass(frozen=True)
class FiniteDPProblem:
    """Finite states and actions: ``reward(p) -> [S, A]``, ``transition(p) -> [A, S, S']``."""

    reward: Fn
    transition: Fn
    discount: str
    state_labels: tuple[str, ...] | None = None
    action_labels: tuple[str, ...] | None = None
