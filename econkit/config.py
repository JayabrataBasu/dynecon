"""Runtime configuration for backend dispatch and reproducibility."""

from dataclasses import dataclass
from typing import Literal

BackendName = Literal["numpy", "numba", "jax", "c_ext", "rust"]


@dataclass
class Config:
    """Global runtime config for EconKit.

    The active backend determines which registered numerical implementation
    the dispatcher routes to at runtime.
    """

    active_backend: BackendName = "numpy"
    strict_backend: bool = False

    def set_backend(self, backend: BackendName) -> None:
        self.active_backend = backend


config = Config()
