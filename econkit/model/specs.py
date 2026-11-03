"""Model specification objects for the EconKit DSL.

Economics note:
Parameters, states, controls and shock processes are the primitives of a dynamic
economic model; keeping them declarative separates economic intent from numerics.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError

__all__ = [
    "AR1Shock",
    "Control",
    "IIDShock",
    "MarkovShock",
    "Param",
    "ParamView",
    "ParameterSet",
    "ShockProcess",
    "State",
]


@dataclass(frozen=True)
class Param:
    """Named model parameter (value, optional bounds, free/fixed flag).

    ``transform`` maps the parameter to the real line for estimation: ``"logit"`` for a
    two-sided bound, ``"log"`` for a lower bound only, ``"none"`` for unbounded; ``"auto"``
    picks from ``bounds``. ``prior`` is an :mod:`econkit.estimation.priors` distribution.

    Economics note:
    Parameters map structural primitives such as preferences, technology and policy
    constants into executable model code; ``free`` marks those to be estimated.
    """

    value: float
    bounds: tuple[float, float] | None = None
    free: bool = False
    description: str = ""
    transform: Literal["auto", "none", "log", "logit"] = "auto"
    prior: Any = None


@dataclass(frozen=True)
class AR1Shock:
    """AR(1) shock ``z' = (1 - rho) mean + rho z + eps``, ``eps ~ N(0, sigma_e^2)``.

    ``rho`` and ``sigma_e`` may be numbers or the *name of a parameter* (so they can be
    estimated). With ``log=True`` the state takes values ``exp(z)`` (productivity).
    """

    rho: float | str
    sigma_e: float | str
    n_states: int
    mean: float = 0.0
    log: bool = False
    method: Literal["rouwenhorst", "tauchen"] = "rouwenhorst"


@dataclass(frozen=True)
class MarkovShock:
    """User-supplied finite Markov chain: state values and transition matrix."""

    values: npt.NDArray[np.float64]
    P: npt.NDArray[np.float64]


@dataclass(frozen=True)
class IIDShock:
    """IID shock discretised by quadrature.

    ``kind`` is ``"normal"`` (params ``mu``, ``sigma``), ``"lognormal"`` (same, of the log)
    or ``"uniform"`` (params ``low``, ``high``).
    """

    kind: Literal["normal", "lognormal", "uniform"]
    params: Mapping[str, float] = field(default_factory=dict)
    n_nodes: int = 5


@dataclass(frozen=True)
class State:
    """State variable. A state with ``shock`` is exogenous; otherwise it needs bounds and points.

    ``grid_type`` is ``"uniform"`` or ``"exp"`` (curved, dense near the lower bound, which is
    the borrowing limit for an asset state).
    """

    name: str
    bounds: tuple[float, float] | None = None
    grid_type: str = "uniform"
    n_points: int | None = None
    shock: AR1Shock | MarkovShock | IIDShock | None = None
    grid_curvature: float = 3.0


@dataclass(frozen=True)
class Control:
    """Control (choice) variable; its name keys the solver's policy output."""

    name: str
    bounds: tuple[float | None, float | None] | None = None


@dataclass(frozen=True)
class ShockProcess:
    """Discretised shock: state values and row-stochastic transition matrix."""

    values: npt.NDArray[np.float64]
    P: npt.NDArray[np.float64]


class ParamView:
    """Read-only attribute view of parameter values, passed as ``p`` to template callables."""

    __slots__ = ("_v",)

    def __init__(self, values: Mapping[str, float]) -> None:
        object.__setattr__(self, "_v", dict(values))

    def __getattr__(self, name: str) -> float:
        try:
            return object.__getattribute__(self, "_v")[name]  # type: ignore[no-any-return]
        except KeyError:
            raise AttributeError(f"no parameter named {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("parameters are read-only")

    def as_dict(self) -> dict[str, float]:
        return dict(object.__getattribute__(self, "_v"))


class ParameterSet:
    """Ordered, immutable collection of :class:`Param` with a vector view for estimation."""

    def __init__(self, params: Mapping[str, Param]) -> None:
        self._params: dict[str, Param] = dict(params)

    def __getitem__(self, name: str) -> Param:
        return self._params[name]

    def __contains__(self, name: object) -> bool:
        return name in self._params

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    @property
    def values(self) -> dict[str, float]:
        return {k: p.value for k, p in self._params.items()}

    @property
    def view(self) -> ParamView:
        return ParamView(self.values)

    def names(self, free: bool = True) -> list[str]:
        """Names of free parameters (``free=True``) or of all parameters."""
        return [k for k, p in self._params.items() if p.free or not free]

    def _select(self, names: list[str] | None) -> list[str]:
        sel = self.names(free=True) if names is None else list(names)
        for n in sel:
            if n not in self._params:
                raise ValidationError(f"unknown parameter {n!r}; known: {list(self._params)}")
        return sel

    def vector(self, names: list[str] | None = None) -> npt.NDArray[np.float64]:
        """Values of ``names`` (default: the free parameters) as a vector."""
        return np.array([self._params[n].value for n in self._select(names)], dtype=np.float64)

    def bounds(self, names: list[str] | None = None) -> list[tuple[float, float]]:
        """Bounds of ``names`` (``-inf, inf`` when unbounded)."""
        out = []
        for n in self._select(names):
            b = self._params[n].bounds
            out.append((-np.inf, np.inf) if b is None else (float(b[0]), float(b[1])))
        return out

    def from_vector(self, x: npt.ArrayLike, names: list[str] | None = None) -> ParameterSet:
        """New set with ``names`` (default: free parameters) set from the vector ``x``."""
        sel = self._select(names)
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != (len(sel),):
            raise ValidationError(f"expected a vector of length {len(sel)}, got shape {arr.shape}")
        return self.replace(**{n: float(v) for n, v in zip(sel, arr, strict=True)})

    def replace(self, **overrides: float) -> ParameterSet:
        """New set with the given parameter values overridden."""
        for n in overrides:
            if n not in self._params:
                raise ValidationError(f"unknown parameter {n!r}; known: {list(self._params)}")
        new = dict(self._params)
        for n, v in overrides.items():
            new[n] = replace(new[n], value=float(v))
        return ParameterSet(new)

    def _kind(self, name: str) -> str:
        p = self._params[name]
        if p.transform != "auto":
            return p.transform
        lo, hi = (-np.inf, np.inf) if p.bounds is None else p.bounds
        if np.isfinite(lo) and np.isfinite(hi):
            return "logit"
        return "log" if np.isfinite(lo) else "none"

    def to_unconstrained(
        self, x: npt.ArrayLike, names: list[str] | None = None
    ) -> npt.NDArray[np.float64]:
        """Map parameter values ``x`` of ``names`` to the real line using each transform."""
        sel = self._select(names)
        arr = np.asarray(x, dtype=np.float64)
        out = np.empty(len(sel))
        for i, n in enumerate(sel):
            kind, (lo, hi) = self._kind(n), self.bounds([n])[0]
            if kind == "logit":
                t = (arr[i] - lo) / (hi - lo)
                if not 0.0 < t < 1.0:
                    raise ValidationError(f"{n}={arr[i]} must lie strictly inside ({lo}, {hi})")
                out[i] = np.log(t / (1.0 - t))
            elif kind == "log":
                if not arr[i] > lo:
                    raise ValidationError(f"{n}={arr[i]} must exceed its lower bound {lo}")
                out[i] = np.log(arr[i] - (lo if np.isfinite(lo) else 0.0))
            else:
                out[i] = arr[i]
        return out

    def from_unconstrained(
        self, u: npt.ArrayLike, names: list[str] | None = None
    ) -> npt.NDArray[np.float64]:
        """Inverse of :meth:`to_unconstrained` (always inside the bounds)."""
        sel = self._select(names)
        arr = np.asarray(u, dtype=np.float64)
        out = np.empty(len(sel))
        for i, n in enumerate(sel):
            kind, (lo, hi) = self._kind(n), self.bounds([n])[0]
            if kind == "logit":
                out[i] = lo + (hi - lo) / (1.0 + np.exp(-arr[i]))
            elif kind == "log":
                out[i] = (lo if np.isfinite(lo) else 0.0) + np.exp(arr[i])
            else:
                out[i] = arr[i]
        return out

    def log_jacobian(self, u: npt.ArrayLike, names: list[str] | None = None) -> float:
        """``log |d theta / d u|`` at the unconstrained point ``u`` (for Bayesian densities)."""
        sel = self._select(names)
        arr = np.asarray(u, dtype=np.float64)
        total = 0.0
        for i, n in enumerate(sel):
            kind, (lo, hi) = self._kind(n), self.bounds([n])[0]
            if kind == "logit":
                s = 1.0 / (1.0 + np.exp(-arr[i]))
                total += np.log(hi - lo) + np.log(s) + np.log(1.0 - s)
            elif kind == "log":
                total += arr[i]
        return float(total)
