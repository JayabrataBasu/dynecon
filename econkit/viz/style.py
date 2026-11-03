"""Plot style and lazy matplotlib import.

Economics note:
Consistent, readable figures (colour-blind-safe colours, labelled axes) make results
comparable across models and publication-ready by default.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["econkit_style", "get_pyplot"]

# Okabe-Ito colour-blind-safe cycle
_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def get_pyplot() -> Any:
    """Import ``matplotlib.pyplot`` lazily with an informative error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("plotting needs matplotlib: pip install econkit[viz]") from exc
    return plt


@contextmanager
def econkit_style() -> Iterator[None]:
    """Context manager applying EconKit's plot defaults without touching global rcParams."""
    plt = get_pyplot()
    params = {
        "axes.prop_cycle": plt.cycler(color=_COLORS),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "figure.dpi": 100,
    }
    with plt.rc_context(params):
        yield


def get_axes(ax: Any | None, **fig_kw: Any) -> Any:
    """Return ``ax`` or a new axes created inside the EconKit style."""
    if ax is not None:
        return ax
    plt = get_pyplot()
    with econkit_style():
        _, new_ax = plt.subplots(**fig_kw)
    return new_ax
