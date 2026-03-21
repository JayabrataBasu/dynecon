"""Shared typed aliases and shape-friendly containers."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

GridType = Literal["uniform", "log", "sinh", "chebyshev"]


@dataclass(frozen=True)
class StateSpec:
    name: str
    grid: np.ndarray
    grid_type: GridType = "uniform"


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
