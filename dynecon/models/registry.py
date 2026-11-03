"""Model registry: ``available()``, ``load()`` and ``register()``.

Economics note:
A registry of canonical models lets students and researchers start from a well-understood
benchmark and modify one assumption at a time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dynecon.core.errors import ValidationError
from dynecon.model import Model

__all__ = ["available", "load", "register"]

_BUILDERS: dict[str, Callable[..., Model]] = {}


def register(name: str, builder: Callable[..., Model]) -> None:
    """Register a model builder under ``name`` (the same call the built-ins use)."""
    if name in _BUILDERS:
        raise ValidationError(f"a model named {name!r} is already registered")
    _BUILDERS[name] = builder


def available() -> list[str]:
    """Names of all registered models, sorted."""
    return sorted(_BUILDERS)


def load(name: str, **overrides: Any) -> Model:
    """Build the registered model ``name``; ``overrides`` replace default parameters/options."""
    if name not in _BUILDERS:
        raise ValidationError(f"unknown model {name!r}; available: {available()}")
    return _BUILDERS[name](**overrides)


def split_overrides(
    overrides: dict[str, Any], defaults: dict[str, Any], what: str
) -> dict[str, Any]:
    """Merge ``overrides`` onto ``defaults``; unknown names are an error."""
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValidationError(f"unknown {what} option(s) {unknown}; valid: {sorted(defaults)}")
    return {**defaults, **overrides}
