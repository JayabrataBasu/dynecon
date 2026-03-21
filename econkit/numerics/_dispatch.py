"""Backend dispatch infrastructure for numerical kernels."""

from collections.abc import Callable
from typing import Any

from econkit.config import config
from econkit.numerics._registry import get

_FALLBACK_ORDER = ["numpy"]


def resolve_backend(operation: str, backend: str | None = None) -> tuple[str, Callable[..., Any]]:
    """Resolve a callable implementation for an operation and backend.

    If strict backend mode is disabled, this function falls back to NumPy.
    """
    selected = backend or config.active_backend
    impl = get(operation, selected)
    if impl is not None:
        return selected, impl

    if config.strict_backend:
        raise ValueError(f"No implementation for operation={operation} backend={selected}")

    for fallback in _FALLBACK_ORDER:
        fallback_impl = get(operation, fallback)
        if fallback_impl is not None:
            return fallback, fallback_impl

    raise ValueError(f"No implementation found for operation={operation}")
