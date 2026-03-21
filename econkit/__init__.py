"""EconKit public package entrypoint."""

from econkit.config import config
from econkit.numerics.interpolation import linear_interp_1d

__all__ = ["config", "linear_interp_1d"]
__version__ = "0.1.0.dev0"
