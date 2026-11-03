"""Model validation: collect every problem and report them together.

Economics note:
A mis-specified primitive (an impossible discount factor, a budget constraint with the
wrong shape) must fail loudly at compile time, not as a silent nonsense solution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from econkit.model.problems import (
    DSGEProblem,
    FiniteDPProblem,
    PricingProblem,
    SavingProblem,
    Vars,
)
from econkit.model.specs import ParameterSet, ParamView, ShockProcess

if TYPE_CHECKING:
    from econkit.model.compiled import ModelSpec

__all__ = ["validate_model"]


def _call(errors: list[str], field: str, fn: Callable[[], Any]) -> np.ndarray | None:
    try:
        out = np.asarray(fn(), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - user callables can raise anything
        errors.append(f"{field}: callable raised {type(exc).__name__}: {exc}")
        return None
    if not np.all(np.isfinite(out)):
        errors.append(f"{field}: returned non-finite values")
        return None
    return out


def _check_shape(
    errors: list[str], field: str, out: np.ndarray | None, shape: tuple[int, ...]
) -> None:
    if out is None:
        return
    try:
        np.broadcast_to(out, shape)
    except ValueError:
        errors.append(f"{field}: returned shape {out.shape}, expected broadcastable to {shape}")


def _check_saving_references(
    sp: SavingProblem,
    spec: ModelSpec,
    params: ParameterSet,
    grids: Mapping[str, np.ndarray],
    shocks: Mapping[str, ShockProcess],
    errors: list[str],
) -> None:
    if sp.asset not in spec.state_map:
        errors.append(f"saving.asset: no state named {sp.asset!r}")
    elif sp.asset not in grids:
        errors.append(f"saving.asset: state {sp.asset!r} has no grid (needs bounds and n_points)")
    if sp.shock is not None and sp.shock not in shocks:
        errors.append(f"saving.shock: state {sp.shock!r} does not carry a shock process")
    if sp.control not in spec.control_map:
        errors.append(f"saving.control: no control named {sp.control!r}")
    if sp.discount not in params:
        errors.append(f"saving.discount: no parameter named {sp.discount!r}")
    elif not 0.0 < params[sp.discount].value < 1.0:
        errors.append(f"saving.discount: parameter {sp.discount!r} must lie in (0, 1)")


def _check_saving_callables(
    sp: SavingProblem,
    params: ParameterSet,
    grids: Mapping[str, np.ndarray],
    shocks: Mapping[str, ShockProcess],
    errors: list[str],
) -> None:
    p: ParamView = params.view
    a = grids[sp.asset]
    z = shocks[sp.shock].values if sp.shock is not None else np.array([1.0])
    shape = (a.shape[0], z.shape[0])
    A, Z = a[:, None], z[None, :]
    res = _call(errors, "resources", lambda: sp.resources(A, Z, p))
    _check_shape(errors, "resources", res, shape)
    if sp.marginal_return is not None:
        mr_fn = sp.marginal_return
        mr = _call(errors, "marginal_return", lambda: mr_fn(A, Z, p))
        _check_shape(errors, "marginal_return", mr, shape)
    c = np.linspace(0.5, 1.5, 5)
    u = _call(errors, "utility", lambda: sp.utility(c, p))
    if u is not None and u.shape != c.shape:
        errors.append(f"utility: returned shape {u.shape}, expected {c.shape}")
    if sp.marginal_utility is not None:
        mu_fn = sp.marginal_utility
        mu = _call(errors, "marginal_utility", lambda: mu_fn(c, p))
        if mu is not None and mu.shape != c.shape:
            errors.append(f"marginal_utility: returned shape {mu.shape}, expected {c.shape}")
        if mu is not None and sp.inverse_marginal_utility is not None:
            inv_fn = sp.inverse_marginal_utility
            inv = _call(errors, "inverse_marginal_utility", lambda: inv_fn(mu, p))
            if inv is not None and (inv.shape != c.shape or not np.allclose(inv, c, rtol=1e-8)):
                errors.append("inverse_marginal_utility: does not invert marginal_utility")
    partial = [
        n
        for n in ("marginal_utility", "inverse_marginal_utility", "marginal_return")
        if getattr(sp, n) is None
    ]
    if 0 < len(partial) < 3:
        errors.append(
            "saving: EGM needs marginal_utility, inverse_marginal_utility and marginal_return "
            f"together; missing {partial}"
        )


def _validate_saving(
    sp: SavingProblem,
    spec: ModelSpec,
    params: ParameterSet,
    grids: Mapping[str, np.ndarray],
    shocks: Mapping[str, ShockProcess],
    errors: list[str],
) -> None:
    n_before = len(errors)
    _check_saving_references(sp, spec, params, grids, shocks, errors)
    if len(errors) == n_before:
        _check_saving_callables(sp, params, grids, shocks, errors)


def _validate_finite_dp(fp: FiniteDPProblem, params: ParameterSet, errors: list[str]) -> None:
    p = params.view
    if fp.discount not in params:
        errors.append(f"finite_dp.discount: no parameter named {fp.discount!r}")
    elif not 0.0 < params[fp.discount].value < 1.0:
        errors.append(f"finite_dp.discount: parameter {fp.discount!r} must lie in (0, 1)")
    R = _call(errors, "reward", lambda: fp.reward(p))
    T = _call(errors, "transition", lambda: fp.transition(p))
    if R is not None and R.ndim != 2:
        errors.append(f"reward: expected shape [state, action], got {R.shape}")
    if T is not None:
        if T.ndim != 3 or T.shape[1] != T.shape[2]:
            errors.append(f"transition: expected shape [action, state, next_state], got {T.shape}")
        elif np.any(T < -1e-12) or not np.allclose(T.sum(axis=2), 1.0, atol=1e-10):
            errors.append("transition: rows must be non-negative and sum to one")
        elif R is not None and R.ndim == 2 and R.shape != (T.shape[1], T.shape[0]):
            errors.append(f"reward {R.shape} and transition {T.shape} shapes are inconsistent")


def _validate_dsge(dp: DSGEProblem, params: ParameterSet, errors: list[str]) -> None:
    p = params.view
    names = dp.variables
    if len(set(names)) != len(names):
        errors.append("dsge.variables: names must be unique")
        return
    if tuple(names[: len(dp.states)]) != tuple(dp.states):
        errors.append("dsge.states: predetermined states must come first in `variables`, in order")
    bad = [v for v in dp.log_variables if v not in names]
    if bad:
        errors.append(f"dsge.log_variables: unknown variables {bad}")
    if dp.steady_state is None and dp.guess is None:
        errors.append("dsge: provide `steady_state(p)` or a `guess(p)` for the numerical solver")
        return
    start = _call(errors, "steady_state/guess", lambda: (dp.steady_state or dp.guess)(p))  # type: ignore[misc]
    if start is None:
        return
    if start.shape != (len(names),):
        errors.append(f"dsge.steady_state/guess: expected {len(names)} values, got {start.shape}")
        return
    res = _call(
        errors,
        "equations",
        lambda: dp.equations(Vars(names, start), Vars(names, start), p),
    )
    if res is not None and res.shape != (len(names),):
        errors.append(f"equations: need {len(names)} residuals (one per variable), got {res.shape}")
    load = _call(errors, "shock_loadings", lambda: dp.shock_loadings(p))
    if load is not None and load.shape != (len(dp.states), len(dp.shocks)):
        errors.append(
            f"shock_loadings: expected shape {(len(dp.states), len(dp.shocks))}, got {load.shape}"
        )


def _validate_pricing(
    pp: PricingProblem,
    params: ParameterSet,
    shocks: Mapping[str, ShockProcess],
    errors: list[str],
) -> None:
    if pp.shock not in shocks:
        errors.append(f"pricing.shock: state {pp.shock!r} does not carry a shock process")
        return
    if pp.discount not in params:
        errors.append(f"pricing.discount: no parameter named {pp.discount!r}")
    elif not 0.0 < params[pp.discount].value < 1.0:
        errors.append(f"pricing.discount: parameter {pp.discount!r} must lie in (0, 1)")
    y = shocks[pp.shock].values
    mu = _call(errors, "marginal_utility", lambda: pp.marginal_utility(y, params.view))
    if mu is not None and (mu.shape != y.shape or np.any(mu <= 0.0)):
        errors.append("marginal_utility: must be positive with the same shape as the dividends")


def validate_model(
    spec: ModelSpec,
    params: ParameterSet,
    grids: Mapping[str, np.ndarray],
    shocks: Mapping[str, ShockProcess],
) -> list[str]:
    """Return every problem found with a model (empty list when valid)."""
    errors: list[str] = []
    for name in params:
        prm = params[name]
        if prm.bounds is not None and not prm.bounds[0] <= prm.value <= prm.bounds[1]:
            errors.append(f"parameter {name!r}={prm.value} outside bounds {prm.bounds}")
    if spec.problem is None:
        errors.append("no problem set: call set_problem('saving' | 'finite_dp', ...)")
    elif isinstance(spec.problem, SavingProblem):
        _validate_saving(spec.problem, spec, params, grids, shocks, errors)
    elif isinstance(spec.problem, PricingProblem):
        _validate_pricing(spec.problem, params, shocks, errors)
    elif isinstance(spec.problem, DSGEProblem):
        _validate_dsge(spec.problem, params, errors)
    else:
        _validate_finite_dp(spec.problem, params, errors)
    return errors
