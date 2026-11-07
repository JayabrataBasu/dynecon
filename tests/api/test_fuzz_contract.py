"""Property-based contract: bad numeric input gives a Dynecon error or finite output, never a raw
exception or silent NaN/inf.

Each target is called with generated arguments that mix valid values with NaN, infinities, extreme
magnitudes, zero, negatives, non-integer counts, empty and ragged shapes. Outcomes allowed:
a ``DyneconError`` (a clear refusal), a ``MemoryError`` (an honest resource limit) or a result whose
float arrays are all finite. Examples are derandomised (same cases every run); set
``DYNECON_FUZZ_EXAMPLES`` to explore more.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from dynecon import models, solve
from dynecon.core.errors import DyneconError
from dynecon.data import baxter_king, hamilton_filter, hp_filter
from dynecon.estimation import hmc, kalman_filter, metropolis
from dynecon.numerics import (
    MarkovChain,
    cartesian,
    exp_grid,
    fixed_point,
    gauss_chebyshev,
    gauss_hermite_normal,
    gauss_legendre,
    linear_interp_1d,
    lognormal_nodes,
    mvnormal_nodes,
    rouwenhorst,
    simulate_markov,
    stationary_distribution,
    tauchen,
    trapezoid_rule,
    uniform_grid,
)
from dynecon.simulation import gini, lorenz_curve, quantile
from dynecon.solvers.lq import LQ, solve_discrete_riccati
from dynecon.solvers.perturbation.klein import klein_solve

EXAMPLES = int(os.environ.get("DYNECON_FUZZ_EXAMPLES", "60"))

# ---- value strategies -------------------------------------------------------------------------
wild = st.floats(allow_nan=True, allow_infinity=True)  # includes nan, inf, huge, tiny
near_unit = st.one_of(wild, st.floats(-1.2, 1.2))  # correlations, discount factors
positive = st.one_of(wild, st.floats(1e-3, 5.0))  # scales, tolerances
SPECIAL_COUNTS = [float("nan"), float("inf"), -float("inf"), 2.5, 1e300, 1e9, 1e15, "3", None]
counts = st.one_of(st.integers(-2, 25), st.sampled_from(SPECIAL_COUNTS))
real = st.one_of(wild, st.floats(-3.0, 3.0))


def vec(max_len: int = 8) -> st.SearchStrategy[Any]:
    return hnp.arrays(np.float64, st.integers(0, max_len), elements=real)


def square(max_n: int = 4) -> st.SearchStrategy[Any]:
    return st.integers(0, max_n).flatmap(lambda n: hnp.arrays(np.float64, (n, n), elements=real))


def stochastic() -> st.SearchStrategy[Any]:
    valid = st.integers(1, 5).flatmap(
        lambda n: hnp.arrays(np.float64, (n, n), elements=st.floats(0.01, 1.0)).map(
            lambda a: a / a.sum(axis=1, keepdims=True)
        )
    )
    return st.one_of(valid, valid, square())


@st.composite
def kalman_args(draw: st.DrawFn) -> tuple[Any, ...]:
    na, ny, t = draw(st.integers(1, 3)), draw(st.integers(1, 2)), draw(st.integers(0, 6))

    def m(r: int, c: int) -> Any:
        return draw(hnp.arrays(np.float64, (r, c), elements=real))

    y = draw(hnp.arrays(np.float64, (t, ny), elements=real))
    return (
        y,
        m(ny, na),
        m(ny, ny),
        m(na, na),
        m(na, na),
        draw(hnp.arrays(np.float64, na, elements=real)),
        m(na, na),
    )


@st.composite
def lq_args(draw: st.DrawFn) -> tuple[Any, ...]:
    n, k = draw(st.integers(1, 3)), draw(st.integers(1, 2))

    def m(r: int, c: int) -> Any:
        return draw(hnp.arrays(np.float64, (r, c), elements=real))

    return (m(k, k), m(n, n), m(n, n), m(n, k), draw(near_unit))


@st.composite
def household_args(draw: st.DrawFn) -> dict[str, Any]:
    return {
        "beta": draw(near_unit),
        "sigma": draw(positive),
        "r": draw(near_unit),
        "rho": draw(near_unit),
        "sigma_e": draw(positive),
        "a_max": draw(positive),
        "n_a": draw(st.one_of(counts, st.integers(3, 25))),
        "n_z": draw(st.one_of(counts, st.integers(2, 4))),
    }


def _hmc_call(
    n_draws: Any, n_chains: Any, n_warmup: Any, init_scale: Any, target_accept: Any
) -> Any:
    return hmc(
        lambda x: (-0.5 * x @ x, -x),
        np.zeros(2),
        n_draws,
        n_chains=n_chains,
        n_warmup=n_warmup,
        init_scale=init_scale,
        target_accept=target_accept,
        rng=0,
    )


def _metropolis_call(n_draws: Any, n_chains: Any, burn_in: Any, init_scale: Any) -> Any:
    return metropolis(
        lambda x: -0.5 * x @ x,
        np.zeros(2),
        n_draws,
        n_chains=n_chains,
        burn_in=burn_in,
        init_scale=init_scale,
        rng=0,
    )


def _household(method: str) -> Callable[[dict[str, Any]], Any]:
    def run(kw: dict[str, Any]) -> Any:
        return solve(
            models.load("consumption_savings", **kw).compile(), method=method, max_iter=200
        )

    return run


def _star(f: Callable[..., Any]) -> Callable[[tuple[Any, ...]], Any]:
    return lambda args: f(*args)


T = st.tuples
TARGETS: dict[str, tuple[Callable[[Any], Any], st.SearchStrategy[Any]]] = {
    "tauchen": (_star(tauchen), T(near_unit, positive, counts, positive)),
    "rouwenhorst": (_star(rouwenhorst), T(near_unit, positive, counts)),
    "gauss_hermite_normal": (_star(gauss_hermite_normal), T(counts)),
    "gauss_legendre": (_star(gauss_legendre), T(counts, real, real)),
    "gauss_chebyshev": (_star(gauss_chebyshev), T(counts, real, real)),
    "trapezoid_rule": (_star(trapezoid_rule), T(counts, real, real)),
    "mvnormal_nodes": (_star(mvnormal_nodes), T(st.integers(-1, 4), vec(3), square(3))),
    "lognormal_nodes": (_star(lognormal_nodes), T(st.integers(-1, 4), vec(3), square(3))),
    "cartesian": (_star(cartesian), st.lists(vec(), min_size=0, max_size=3).map(tuple)),
    "exp_grid": (_star(exp_grid), T(real, real, counts, real)),
    "uniform_grid": (_star(uniform_grid), T(real, real, counts)),
    "stationary_distribution": (_star(stationary_distribution), T(stochastic())),
    "simulate_markov": (_star(simulate_markov), T(stochastic(), counts)),
    "MarkovChain.period": (lambda a: MarkovChain(a[0]).period, T(stochastic())),
    "MarkovChain.classes": (lambda a: MarkovChain(a[0]).communication_classes, T(stochastic())),
    "MarkovChain.stationary": (
        lambda a: MarkovChain(a[0]).stationary_distributions,
        T(stochastic()),
    ),
    "linear_interp_1d": (_star(linear_interp_1d), T(vec(), vec(), vec())),
    "fixed_point": (
        lambda a: fixed_point(lambda x: a[0] * x + a[1], a[2], tol=a[3], max_iter=a[4]),
        T(real, real, vec(), positive, counts),
    ),
    "hp_filter": (_star(hp_filter), T(vec(30), positive)),
    "baxter_king": (_star(baxter_king), T(vec(40), counts, counts, counts)),
    "hamilton_filter": (_star(hamilton_filter), T(vec(30), counts, counts)),
    "gini": (_star(gini), T(vec())),
    "gini_weighted": (_star(gini), T(vec(), vec())),
    "lorenz_curve": (_star(lorenz_curve), T(vec(), vec())),
    "quantile": (_star(quantile), T(vec(), vec(), real)),
    "kalman_filter": (_star(kalman_filter), kalman_args()),
    "LQ.stationary_values": (
        lambda a: LQ(a[0], a[1], a[2], a[3], beta=a[4]).stationary_values(),
        lq_args(),
    ),
    "solve_discrete_riccati": (
        lambda a: solve_discrete_riccati(a[2], a[3], a[1], a[0], beta=a[4]),
        lq_args(),
    ),
    "klein_solve": (lambda a: klein_solve(*a)[:2], T(square(3), square(3), st.integers(-1, 4))),
    "hmc": (_star(_hmc_call), T(counts, counts, counts, positive, near_unit)),
    "metropolis": (_star(_metropolis_call), T(counts, counts, counts, positive)),
    "household_egm": (_household("egm"), household_args()),
    "household_vfi": (_household("vfi"), household_args()),
}


def _float_arrays(x: Any, depth: int = 0) -> list[np.ndarray]:
    """Every float array reachable from a result (tuples, dicts, dataclass or object attributes)."""
    if x is None or depth > 3 or isinstance(x, (str, bytes, bool)):
        return []
    if isinstance(x, (float, int, np.floating, np.integer, np.ndarray)):
        a = np.asarray(x)
        return [a] if a.dtype.kind == "f" else []
    if isinstance(x, dict):
        return [a for v in x.values() for a in _float_arrays(v, depth + 1)]
    if isinstance(x, (tuple, list)):
        return [a for v in x for a in _float_arrays(v, depth + 1)]
    names = list(getattr(x, "__dataclass_fields__", {})) or list(getattr(x, "__dict__", {}))
    skip = {"model", "spec", "config", "meta", "metadata", "provenance", "household_solution"}
    return [
        a for n in names if n not in skip for a in _float_arrays(getattr(x, n, None), depth + 1)
    ]


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_clear_error_or_finite_output(name: str) -> None:
    fn, strategy = TARGETS[name]

    @settings(
        max_examples=EXAMPLES,
        deadline=None,
        derandomize=True,
        database=None,
        suppress_health_check=list(HealthCheck),
    )
    @given(strategy)
    def run(args: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = fn(args)
            except (DyneconError, MemoryError):
                return
        for a in _float_arrays(out):
            assert np.all(np.isfinite(a)), f"{name}{args!r} returned non-finite values"

    run()
