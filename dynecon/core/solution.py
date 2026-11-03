"""The one result type returned by every solver.

Economics note:
A solution bundles the policy and value functions with the model that produced them,
convergence diagnostics and provenance, so any downstream simulation or estimate can
be traced back to exact primitives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from dynecon.model.compiled import CompiledModel

__all__ = ["Solution"]

Array = npt.NDArray[np.float64]


@dataclass
class Solution:
    """Result of :func:`dynecon.solve`.

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

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation (arrays as lists, NaN/inf encoded); the model is excluded."""
        from dynecon.utils.serialization import encode

        fields = {
            "method": self.method,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
            "elapsed": self.elapsed,
            "backend": self.backend,
            "model_name": self.model_name,
            "params": self.params,
            "grids": self.grids,
            "policy": self.policy,
            "value": self.value,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }
        return {k: encode(v) for k, v in fields.items()}

    def save(self, path: str | Path) -> Path:
        """Write arrays to ``<path>.npz`` and everything else to ``<path>.json``."""
        import json

        from dynecon.utils.serialization import encode

        base = Path(path)
        arrays: dict[str, Any] = {f"grid__{k}": v for k, v in self.grids.items()}
        arrays |= {f"policy__{k}": np.asarray(v) for k, v in self.policy.items()}
        if self.value is not None:
            arrays["value"] = self.value
        np.savez_compressed(base.with_suffix(".npz"), **arrays)
        scalar = {
            k: encode(v) for k, v in self.to_dict().items() if k not in ("grids", "policy", "value")
        }
        base.with_suffix(".json").write_text(json.dumps(scalar, indent=2), encoding="utf-8")
        return base

    @classmethod
    def load(cls, path: str | Path, model: CompiledModel | None = None) -> Solution:
        """Load a solution written by :meth:`save`; pass ``model`` to re-attach it."""
        import json

        from dynecon.utils.serialization import decode

        base = Path(path)
        meta = decode(json.loads(base.with_suffix(".json").read_text(encoding="utf-8")))
        with np.load(base.with_suffix(".npz")) as data:
            grids = {k[6:]: data[k] for k in data.files if k.startswith("grid__")}
            policy = {k[8:]: data[k] for k in data.files if k.startswith("policy__")}
            value = data["value"] if "value" in data.files else None
        return cls(
            method=meta["method"],
            converged=meta["converged"],
            iterations=meta["iterations"],
            residual=meta["residual"],
            elapsed=meta["elapsed"],
            backend=meta["backend"],
            model_name=meta["model_name"],
            params=meta["params"],
            grids=grids,
            policy=policy,
            value=value,
            diagnostics=meta["diagnostics"],
            metadata=meta["metadata"],
            model=model,
        )

    def policy_function(self, name: str) -> Callable[[Any, Any], Array]:
        """Interpolating callable ``f(assets, shock_index)`` for a ``saving`` policy.

        Linear interpolation along the asset grid (clamped at the ends) for each discrete
        shock state, vectorised over broadcast arguments.

        Economics note:
        Evaluating the policy off the grid lets simulated agents hold any asset level,
        not just grid points.
        """
        from dynecon.model.problems import SavingProblem
        from dynecon.numerics.interpolation import linear_interp_1d

        prob = None if self.model is None else self.model.problem
        if not isinstance(prob, SavingProblem):
            raise ValueError("policy_function needs a solution of a 'saving' problem")
        if name not in self.policy:
            raise ValueError(f"no policy named {name!r}; available: {sorted(self.policy)}")
        grid = self.grids[prob.asset]
        values = np.asarray(self.policy[name], dtype=np.float64)

        def f(assets: Any, shock_index: Any = 0) -> Array:
            a, j = np.broadcast_arrays(
                np.asarray(assets, dtype=np.float64), np.asarray(shock_index, dtype=np.int64)
            )
            out = np.empty(a.shape, dtype=np.float64)
            for jj in np.unique(j):
                mask = j == jj
                out[mask] = linear_interp_1d(a[mask], grid, values[:, jj])
            return out

        return f

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
