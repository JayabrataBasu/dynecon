"""Numerical kernels and backend dispatch APIs."""

from dynecon.numerics.discretization import rouwenhorst, tauchen
from dynecon.numerics.fixedpoint import FixedPointResult, fixed_point
from dynecon.numerics.grids import cartesian, exp_grid, make_grid, uniform_grid
from dynecon.numerics.integration import (
    gauss_chebyshev,
    gauss_hermite_normal,
    gauss_legendre,
    lognormal_nodes,
    mvnormal_nodes,
    tensor_rule,
    trapezoid_rule,
)
from dynecon.numerics.interpolation import linear_interp_1d
from dynecon.numerics.jacobian import numerical_jacobian
from dynecon.numerics.markov import (
    MarkovChain,
    markov_paths,
    simulate_markov,
    stationary_distribution,
)

__all__ = [
    "FixedPointResult",
    "cartesian",
    "gauss_chebyshev",
    "lognormal_nodes",
    "mvnormal_nodes",
    "tensor_rule",
    "trapezoid_rule",
    "MarkovChain",
    "exp_grid",
    "fixed_point",
    "gauss_hermite_normal",
    "gauss_legendre",
    "linear_interp_1d",
    "make_grid",
    "markov_paths",
    "numerical_jacobian",
    "rouwenhorst",
    "simulate_markov",
    "stationary_distribution",
    "tauchen",
    "uniform_grid",
]
