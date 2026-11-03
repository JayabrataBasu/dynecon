"""The one result type returned by every solver.

Economics note:
A solution bundles the policy and value functions with the model that produced them,
convergence diagnostics and provenance, so any downstream simulation or estimate can
be traced back to exact primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from econkit.model.compiled import CompiledModel

__all__ = ["Solution"]

Array = npt.NDArray[np.float64]


@dataclass
class Solution:
    """Result of :func:`econkit.solve`.

    ``policy`` maps names to arrays shaped ``(n_assets, n_shocks)`` for ``saving`` problems:
    the control (e.g. ``"consumption"``), next-period assets ``"<asset>_next"`` and, for grid
    methods, its index ``"<asset>_next_index"``. ``value`` is ``None`` for methods that do
    not compute it (EGM).
    """

    method: str
    converged: bool
    iterations: int
    residual: float
    elapsed: float
    backend: str
    model_name: str | None
    params: dict[str, float]
    grids: dict[str, Array]
    policy: dict[str, npt.NDArray[Any]]
    value: Array | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    model: CompiledModel | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Human-readable report."""
        lines = [
            f"Solution of '{self.model_name}' by {self.method} (backend: {self.backend})",
            f"  converged: {self.converged} after {self.iterations} iterations "
            f"(residual {self.residual:.3e}, {self.elapsed:.3f}s)",
        ]
        for key in sorted(self.diagnostics):
            val = self.diagnostics[key]
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.4g}")
            elif isinstance(val, (int, bool)):
                lines.append(f"  {key}: {val}")
        fp = self.metadata.get("fingerprint")
        if fp:
            lines.append(f"  model fingerprint: {str(fp)[:12]}")
        return "\n".join(lines)
