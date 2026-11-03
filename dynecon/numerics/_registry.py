"""Dispatcher registry for backend-specific numerical kernels."""

from collections.abc import Callable
from typing import Any

RegistryKey = tuple[str, str]
RegistryImpl = Callable[..., Any]

_REGISTRY: dict[RegistryKey, RegistryImpl] = {}


def register(operation: str, backend: str, impl: RegistryImpl) -> None:
    """Register a numerical kernel implementation."""
    key = (operation, backend)
    if key in _REGISTRY:
        raise ValueError(f"Implementation already registered for {key}")
    _REGISTRY[key] = impl


def get(operation: str, backend: str) -> RegistryImpl | None:
    return _REGISTRY.get((operation, backend))


def list_registered(operation: str | None = None) -> dict[RegistryKey, RegistryImpl]:
    if operation is None:
        return dict(_REGISTRY)
    return {k: v for k, v in _REGISTRY.items() if k[0] == operation}
