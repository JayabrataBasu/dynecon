"""Problem templates: the contract between a model and a family of solvers.

Economics note:
A template names exactly the ingredients a solver needs (preferences, budget
constraint, returns), so nothing in a solver ever calls an arbitrary equation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["DSGEProblem", "FiniteDPProblem", "PricingProblem", "SavingProblem", "Vars"]

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


class Vars:
    """Name-addressable view of a vector of model variables: ``y.k``, ``y.c``.

    Economics note:
    Writing equations as ``y.c ** -sigma`` instead of ``y[2] ** -sigma`` keeps a DSGE
    system readable and independent of the variable ordering.
    """

    __slots__ = ("_idx", "_v")

    def __init__(self, names: tuple[str, ...], values: Any) -> None:
        object.__setattr__(self, "_idx", {n: i for i, n in enumerate(names)})
        object.__setattr__(self, "_v", values)

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "_v")[object.__getattribute__(self, "_idx")[name]]
        except KeyError:
            raise AttributeError(f"no variable named {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("variables are read-only")


@dataclass(frozen=True)
class DSGEProblem:
    """Linear-rational-expectations template ``E_t f(y_{t+1}, y_t; p) = 0``.

    ``variables`` lists predetermined ``states`` first, then jump variables. ``equations(yp,
    y, p)`` receives :class:`Vars` of next-period and current levels and returns one residual
    per variable. ``shock_loadings(p)`` is the ``(n_states, n_shocks)`` matrix mapping
    structural shocks into the state innovations. The steady state comes from
    ``steady_state(p)`` (levels) or is solved numerically from ``guess(p)``.
    """

    variables: tuple[str, ...]
    states: tuple[str, ...]
    shocks: tuple[str, ...]
    equations: Fn
    shock_loadings: Fn
    steady_state: Fn | None = None
    guess: Fn | None = None
    log_variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class PricingProblem:
    """Endowment-economy asset pricing: consumption equals the dividend of a Lucas tree.

    ``shock`` names the state carrying the dividend process (values are the dividend),
    ``discount`` the discount-factor parameter and ``marginal_utility(c, p)`` the household's
    marginal utility. The price satisfies ``p(y) u'(y) = beta E[u'(y') (p(y') + y')]``.
    """

    shock: str
    discount: str
    marginal_utility: Fn
