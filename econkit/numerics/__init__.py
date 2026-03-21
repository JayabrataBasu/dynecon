"""Numerical kernels and backend dispatch APIs."""

from econkit.numerics.discretization import rouwenhorst, tauchen
from econkit.numerics.integration import (
    expectation_discrete,
    gauss_hermite_normal,
    gauss_laguerre_exp,
)
from econkit.numerics.optimize import brent_max, golden_section_max
from econkit.numerics.interpolation import linear_interp_1d

__all__ = [
    "linear_interp_1d",
    "tauchen",
    "rouwenhorst",
    "gauss_hermite_normal",
    "gauss_laguerre_exp",
    "expectation_discrete",
    "golden_section_max",
    "brent_max",
]
