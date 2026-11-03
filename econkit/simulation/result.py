"""Container for simulated paths with moment accessors.

Economics note:
A simulation result keeps the paths together with the seed and the model fingerprint,
so every simulated moment can be reproduced and attributed to exact primitives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.simulation import moments as mom

__all__ = ["SimulationResult"]

Array = npt.NDArray[Any]

_MOMENT = re.compile(r"^(mean|std|var|autocorr|p(?:\d+(?:\.\d+)?))_(.+)$")


@dataclass(frozen=True)
class SimulationResult:
    """Simulated panel: ``paths[name]`` has shape ``(n_periods, n_agents)``."""

    paths: dict[str, Array]
    n_periods: int
    n_agents: int
    seed_entropy: int | None
    solution_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def _path(self, name: str) -> Array:
        if name not in self.paths:
            raise ValidationError(
                f"no simulated variable {name!r}; available: {sorted(self.paths)}"
            )
        return self.paths[name]

    def mean(self, name: str) -> float:
        """Mean of variable ``name`` pooled over periods and agents."""
        return mom.sample_mean(self._path(name))

    def moments(self, names: list[str]) -> dict[str, float]:
        """Moments by name: ``mean_x``, ``std_x``, ``var_x``, ``autocorr_x``, ``p50_x`` ..."""
        out: dict[str, float] = {}
        for name in names:
            m = _MOMENT.match(name)
            if m is None:
                raise ValidationError(f"cannot parse moment name {name!r}; use e.g. 'mean_assets'")
            kind, var = m.group(1), m.group(2)
            x = self._path(var).astype(np.float64)
            if kind == "mean":
                out[name] = mom.sample_mean(x)
            elif kind == "std":
                out[name] = mom.sample_std(x)
            elif kind == "var":
                out[name] = mom.sample_std(x) ** 2
            elif kind == "autocorr":
                out[name] = mom.autocorrelation(x, 1)
            else:
                out[name] = mom.percentile(x, float(kind[1:]))
        return out

    def summary(self) -> str:
        """Mean and standard deviation of every float variable."""
        lines = [f"SimulationResult: {self.n_periods} periods x {self.n_agents} agents"]
        for name, arr in sorted(self.paths.items()):
            if np.issubdtype(arr.dtype, np.floating):
                lines.append(f"  {name:<20s} mean {np.mean(arr):12.5g}   std {np.std(arr):12.5g}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation (arrays as lists)."""
        return {
            "paths": {k: v.tolist() for k, v in self.paths.items()},
            "n_periods": self.n_periods,
            "n_agents": self.n_agents,
            "seed_entropy": self.seed_entropy,
            "solution_fingerprint": self.solution_fingerprint,
        }

    def to_dataframe(self) -> Any:
        """Long-format ``pandas.DataFrame`` with ``period`` and ``agent`` (needs pandas)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised only without pandas
            raise ImportError("to_dataframe needs pandas: pip install econkit[data]") from exc
        t, n = self.n_periods, self.n_agents
        data: dict[str, Array] = {
            "period": np.repeat(np.arange(t), n),
            "agent": np.tile(np.arange(n), t),
        }
        for name, arr in self.paths.items():
            data[name] = arr.reshape(-1)
        return pd.DataFrame(data)
