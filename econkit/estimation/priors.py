"""Prior distributions for Bayesian estimation (closed-form log densities).

Economics note:
Priors encode what is believed about structural parameters before seeing data, for
example that a discount factor lies in (0, 1) or that a persistence is likely high.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from econkit.core.errors import ValidationError

__all__ = ["Beta", "Gamma", "InverseGamma", "Normal", "Prior", "Uniform"]


class Prior:
    """Base class: ``logpdf``, ``sample`` and ``support`` (lower, upper)."""

    @property
    def support(self) -> tuple[float, float]:
        raise NotImplementedError

    def _logpdf_inside(self, x: float) -> float:
        raise NotImplementedError

    def logpdf(self, x: float) -> float:
        """Log density (``-inf`` outside the support)."""
        lo, hi = self.support
        if not lo < x < hi:
            return float("-inf")
        return self._logpdf_inside(float(x))

    def sample(self, rng: np.random.Generator) -> float:
        """One draw."""
        raise NotImplementedError


@dataclass(frozen=True)
class Normal(Prior):
    mean: float = 0.0
    std: float = 1.0

    def __post_init__(self) -> None:
        if not self.std > 0:
            raise ValidationError("std must be positive")

    @property
    def support(self) -> tuple[float, float]:
        return (-math.inf, math.inf)

    def _logpdf_inside(self, x: float) -> float:
        z = (x - self.mean) / self.std
        return -0.5 * z * z - math.log(self.std) - 0.5 * math.log(2 * math.pi)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.std))


@dataclass(frozen=True)
class Uniform(Prior):
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValidationError("need high > low")

    @property
    def support(self) -> tuple[float, float]:
        return (self.low, self.high)

    def _logpdf_inside(self, x: float) -> float:
        return -math.log(self.high - self.low)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Beta(Prior):
    a: float = 1.0
    b: float = 1.0

    def __post_init__(self) -> None:
        if not (self.a > 0 and self.b > 0):
            raise ValidationError("a and b must be positive")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def _logpdf_inside(self, x: float) -> float:
        norm = math.lgamma(self.a + self.b) - math.lgamma(self.a) - math.lgamma(self.b)
        return norm + (self.a - 1) * math.log(x) + (self.b - 1) * math.log1p(-x)

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.a, self.b))


@dataclass(frozen=True)
class Gamma(Prior):
    shape: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not (self.shape > 0 and self.scale > 0):
            raise ValidationError("shape and scale must be positive")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, math.inf)

    def _logpdf_inside(self, x: float) -> float:
        return (
            (self.shape - 1) * math.log(x)
            - x / self.scale
            - math.lgamma(self.shape)
            - self.shape * math.log(self.scale)
        )

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.gamma(self.shape, self.scale))


@dataclass(frozen=True)
class InverseGamma(Prior):
    shape: float = 3.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not (self.shape > 0 and self.scale > 0):
            raise ValidationError("shape and scale must be positive")

    @property
    def support(self) -> tuple[float, float]:
        return (0.0, math.inf)

    def _logpdf_inside(self, x: float) -> float:
        return (
            self.shape * math.log(self.scale)
            - math.lgamma(self.shape)
            - (self.shape + 1) * math.log(x)
            - self.scale / x
        )

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.scale / rng.gamma(self.shape, 1.0))
