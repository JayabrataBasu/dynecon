"""Read a Dynare ``model(linear)`` block, solve it and evaluate the Gaussian likelihood.

Economics note:
Most medium-scale DSGE models (Smets-Wouters and its relatives) are published as linear
rational-expectations systems in Dynare syntax. Reading that text directly means the same
equations that were published are the ones estimated, with no retyping.

Supported syntax: ``var``, ``varexo``, equations ``lhs = rhs;`` with ``x(-1)`` and ``x(+1)``
(``x(1)``), parameters supplied as numbers, ``^`` for powers. Shock standard deviations and
parameter values are passed at solve time; derived parameters can be computed with
:func:`evaluate_assignments`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cache
from types import CodeType

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.estimation.kalman import kalman_filter
from dynecon.solvers.perturbation.klein import klein_solve

__all__ = ["LinearDynareModel", "evaluate_assignments", "parse_linear_model"]

Array = npt.NDArray[np.float64]
_SAFE: dict[str, object] = {"__builtins__": {}}
_FUNCS = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt}


@cache
def _code(src: str) -> CodeType:
    return compile(src.strip(), "<equation>", "eval")


class _Lin:
    """Linear expression in ``(variable, lag)`` pairs plus a constant; ``x(-1)`` shifts the lag."""

    def __init__(self, coef: dict[tuple[str, int], float] | None = None, const: float = 0.0):
        self.coef, self.const = dict(coef or {}), const

    def __call__(self, shift: int) -> _Lin:
        return _Lin({(n, lag + int(shift)): v for (n, lag), v in self.coef.items()}, self.const)

    @staticmethod
    def _lift(o: object) -> _Lin:
        return o if isinstance(o, _Lin) else _Lin({}, float(o))  # type: ignore[arg-type]

    def __add__(self, o: object) -> _Lin:
        o = self._lift(o)
        coef = dict(self.coef)
        for key, v in o.coef.items():
            coef[key] = coef.get(key, 0.0) + v
        return _Lin(coef, self.const + o.const)

    __radd__ = __add__

    def __neg__(self) -> _Lin:
        return _Lin({key: -v for key, v in self.coef.items()}, -self.const)

    def __sub__(self, o: object) -> _Lin:
        return self + (-self._lift(o))

    def __rsub__(self, o: object) -> _Lin:
        return self._lift(o) + (-self)

    def __mul__(self, a: object) -> _Lin:
        if isinstance(a, _Lin):
            raise ValidationError("the model is not linear: a product of two variables")
        return _Lin({key: v * float(a) for key, v in self.coef.items()}, self.const * float(a))  # type: ignore[arg-type]

    __rmul__ = __mul__

    def __truediv__(self, a: object) -> _Lin:
        if isinstance(a, _Lin):
            raise ValidationError("the model is not linear: division by a variable")
        return self * (1.0 / float(a))  # type: ignore[arg-type,no-any-return]


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(//|%)[^\n]*", "", text)


def evaluate_assignments(text: str, namespace: dict[str, float]) -> dict[str, float]:
    """Evaluate ``name = expression;`` lines in order (derived parameters), returning the new names.

    Expressions may use earlier names, numbers, ``+ - * / ^`` and ``exp``, ``log``, ``sqrt``.
    """
    ns: dict[str, float] = dict(namespace)
    new: dict[str, float] = {}
    for stmt in _strip(text).split(";"):
        if "=" not in stmt:
            continue
        name, expr = (s.strip() for s in stmt.split("=", 1))
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            raise ValidationError(f"not an assignment: {stmt.strip()!r}")
        try:
            value = float(eval(" ".join(expr.split()).replace("^", "**"), _SAFE, {**_FUNCS, **ns}))  # noqa: S307
        except (NameError, SyntaxError, ZeroDivisionError, TypeError, ValueError) as err:
            raise ValidationError(f"cannot evaluate {name} = {expr.strip()!r}: {err}") from err
        ns[name] = new[name] = value
    return new


@dataclass(frozen=True)
class LinearDynareModel:
    """A parsed linear model: endogenous variables, shocks and equation strings."""

    variables: tuple[str, ...]
    shocks: tuple[str, ...]
    equations: tuple[str, ...]

    def matrices(self, params: dict[str, float]) -> tuple[Array, Array, Array, Array, Array]:
        """``(C0, C1, C2, D, const)``: ``C0 E y' + C1 y + C2 y_{-1} + D e`` equals ``const``."""
        n, m = len(self.variables), len(self.shocks)
        env: dict[str, object] = {v: _Lin({(v, 0): 1.0}) for v in self.variables}
        env.update({e: _Lin({(e, 0): 1.0}) for e in self.shocks})
        env.update(_FUNCS)
        env.update(params)
        vi = {v: i for i, v in enumerate(self.variables)}
        ei = {e: i for i, e in enumerate(self.shocks)}
        C0, C1, C2, D = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, m))
        const = np.zeros(n)
        for r, eq in enumerate(self.equations):
            lhs, rhs = eq.split("=")
            try:
                e = _Lin._lift(eval(_code(lhs), _SAFE, env)) - _Lin._lift(
                    eval(_code(rhs), _SAFE, env)
                )  # noqa: S307
            except NameError as err:
                raise ValidationError(f"equation {r + 1} uses an undefined name: {err}") from err
            for (name, lag), v in e.coef.items():
                if name in ei:
                    D[r, ei[name]] += v
                elif lag in (-1, 0, 1):
                    {1: C0, 0: C1, -1: C2}[lag][r, vi[name]] += v
                else:
                    raise ValidationError(f"{name}({lag:+d}) in equation {r + 1}: only lags -1..+1")
            const[r] = e.const
        return C0, C1, C2, D, const

    def state_space(
        self,
        params: dict[str, float],
        shock_sd: dict[str, float],
        observables: list[str],
    ) -> tuple[Array, Array, Array, Array, Array]:
        """First-order solution y_obs = d + Z a, a' = T a + R e; returns (Z, T, Q, d, P0).

        The state is ``a = [y_{-1}; e]`` reduced to what can reach an observable. ``d`` is the
        steady state of each observable (the constant of its defining equation) and ``P0`` is
        Dynare's ``lik_init=2`` initial covariance (10 on lagged variables, shock variances on
        shocks), provided for models whose stationary covariance is awkward.
        Raises :class:`DeterminacyError` unless the solution exists and is unique.
        """
        unknown = [o for o in observables if o not in self.variables]
        missing = [e for e in self.shocks if e not in shock_sd]
        if unknown or missing:
            raise ValidationError(f"unknown observables {unknown} or missing shock sd {missing}")
        C0, C1, C2, D, const = self.matrices(params)
        n, m = len(self.variables), len(self.shocks)
        N = 2 * n + m
        A, B = np.zeros((N, N)), np.zeros((N, N))
        A[:n, :n] = np.eye(n)  # w = [y_{-1}; eps; y]; predetermined first
        B[:n, n + m :] = np.eye(n)
        A[n : n + m, n : n + m] = np.eye(m)
        A[n + m :, n + m :] = C0
        B[n + m :, n + m :] = -C1
        B[n + m :, :n] = -C2
        B[n + m :, n : n + m] = -D
        F, T, _ = klein_solve(A, B, n + m)
        sd = np.array([shock_sd[e] for e in self.shocks], dtype=np.float64)
        R = np.zeros((n + m, m))
        R[n:, :] = np.diag(sd)
        rows = [self.variables.index(o) for o in observables]
        lhs = [e.split("=")[0].strip() for e in self.equations]
        d = np.array([-const[lhs.index(o)] if o in lhs else 0.0 for o in observables])
        Z = F[rows]
        keep = np.abs(Z).sum(axis=0) > 0  # states that can reach an observable
        while True:
            more = keep | (np.abs(T[keep]).sum(axis=0) > 0)
            if (more == keep).all():
                break
            keep = more
        ix = np.flatnonzero(keep)
        lagged = np.abs(C2).sum(axis=0) > 0
        P0 = np.zeros((n + m, n + m))
        P0[:n, :n] = np.diag(10.0 * lagged)
        P0[n:, n:] = np.diag(sd**2)
        return Z[:, ix], T[np.ix_(ix, ix)], (R @ R.T)[np.ix_(ix, ix)], d, P0[np.ix_(ix, ix)]

    def loglik(
        self,
        data: npt.ArrayLike,
        params: dict[str, float],
        shock_sd: dict[str, float],
        observables: list[str],
        presample: int = 0,
        init: str = "stationary",
    ) -> float:
        """Gaussian log-likelihood of ``data`` (columns ordered as ``observables``, in levels).

        ``init`` is ``"stationary"`` (ergodic covariance) or ``"dynare2"`` (Dynare ``lik_init=2``).
        The first ``presample`` periods update the filter but do not enter the likelihood.
        """
        from scipy.linalg import solve_discrete_lyapunov

        y = np.atleast_2d(np.asarray(data, dtype=np.float64))
        if y.shape[1] != len(observables) or presample < 0 or presample >= y.shape[0]:
            raise ValidationError("data must have one column per observable and exceed presample")
        if init not in ("stationary", "dynare2"):
            raise ValidationError("init must be 'stationary' or 'dynare2'")
        Z, T, Q, d, P0 = self.state_space(params, shock_sd, observables)
        P_init = solve_discrete_lyapunov(T, Q) if init == "stationary" else P0
        H, a0 = np.zeros((len(observables),) * 2), np.zeros(T.shape[0])
        total = kalman_filter(y - d, Z, H, T, Q, a0, P_init).loglik
        if presample:
            total -= kalman_filter((y - d)[:presample], Z, H, T, Q, a0, P_init).loglik
        return float(total)


def parse_linear_model(text: str) -> LinearDynareModel:
    """Parse the ``var``, ``varexo`` and ``model(linear)`` blocks of Dynare text."""
    body = _strip(text)
    try:
        variables = re.search(r"\bvar\s+(.*?);", body, re.S).group(1).split()  # type: ignore[union-attr]
        shocks = re.search(r"\bvarexo\s+(.*?);", body, re.S).group(1).split()  # type: ignore[union-attr]
        block = re.search(r"model\s*\(\s*linear\s*\)\s*;(.*?)\bend\s*;", body, re.S).group(1)  # type: ignore[union-attr]
    except AttributeError as err:
        raise ValidationError("need var, varexo and a model(linear); ... end; block") from err
    eqs = [
        " ".join(e.split()).replace("^", "**")
        for e in block.split(";")
        if "=" in e and not e.strip().startswith("#")
    ]
    if len(eqs) != len(variables):
        raise ValidationError(f"{len(eqs)} equations for {len(variables)} variables")
    return LinearDynareModel(tuple(variables), tuple(shocks), tuple(eqs))
