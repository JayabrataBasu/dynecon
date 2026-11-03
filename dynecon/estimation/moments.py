"""Moment targets and model-implied moments from simulations.

Economics note:
Structural estimation matches a model's simulated moments (means, volatilities, persistence,
inequality) to the same moments in data; the weights say how much each moment matters.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.simulation import moments as mom
from dynecon.simulation.distribution import gini
from dynecon.simulation.result import SimulationResult

__all__ = ["MomentSet", "MomentTarget", "model_moments", "quadratic_moment_loss"]

Array = npt.NDArray[np.float64]

_AUTOCORR = re.compile(r"^autocorr(\d+)_(.+)$")


@dataclass(frozen=True)
class MomentTarget:
    """A named target moment with its weight in a simple diagonal objective."""

    name: str
    value: float
    weight: float = 1.0


def quadratic_moment_loss(simulated: dict[str, float], targets: Sequence[MomentTarget]) -> float:
    """Weighted quadratic distance ``sum_k w_k (m_k - target_k)^2``."""
    if not targets:
        raise ValidationError("targets must contain at least one moment")
    loss = 0.0
    for t in targets:
        if t.name not in simulated:
            raise ValidationError(f"simulated moments missing target: {t.name}")
        loss += t.weight * (simulated[t.name] - t.value) ** 2
    return float(loss)


def model_moments(sim: SimulationResult, names: Sequence[str]) -> Array:
    """Moments of a simulation by name (vector in the order of ``names``).

    Names: ``mean_x``, ``std_x``, ``var_x``, ``autocorr_x``, ``autocorr{k}_x``, ``p{q}_x``,
    ``gini_x`` and the pair moments ``corr:x:y``, ``slope:y:x``.
    """
    out = np.empty(len(names))
    for i, name in enumerate(names):
        if name.startswith("corr:") or name.startswith("slope:"):
            kind, a, b = name.split(":")
            x, y = sim.paths[a], sim.paths[b]
            out[i] = mom.correlation(x, y) if kind == "corr" else mom.regression_slope(x, y)
        elif name.startswith("gini_"):
            out[i] = gini(sim.paths[name[5:]].ravel())
        elif (m := _AUTOCORR.match(name)) is not None:
            out[i] = mom.autocorrelation(sim.paths[m.group(2)], int(m.group(1)))
        else:
            out[i] = sim.moments([name])[name]
    return out


@dataclass(frozen=True)
class MomentSet:
    """Targets (names and data values) and the weights used to compare them with the model.

    Data moments that are NaN are dropped with a note in :attr:`dropped`.
    """

    targets: tuple[MomentTarget, ...]
    dropped: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        keep = tuple(t for t in self.targets if np.isfinite(t.value))
        object.__setattr__(self, "dropped", tuple(t.name for t in self.targets if t not in keep))
        object.__setattr__(self, "targets", keep)
        if not keep:
            raise ValidationError("MomentSet needs at least one finite data moment")

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.targets]

    @property
    def data(self) -> Array:
        return np.array([t.value for t in self.targets], dtype=np.float64)

    @property
    def weights(self) -> Array:
        """Diagonal weight matrix from the target weights."""
        return np.diag([t.weight for t in self.targets]).astype(np.float64)

    def model_moments(self, sim: SimulationResult) -> Array:
        """Model-implied moments, aligned with :attr:`data`."""
        return model_moments(sim, self.names)
