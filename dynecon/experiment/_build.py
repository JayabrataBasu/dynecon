"""Rebuild models from a serialisable ``build`` description (never from pickled code)."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from dynecon.core.errors import ValidationError
from dynecon.model import Model
from dynecon.models import available, load

__all__ = ["build_model", "normalise_build"]


def normalise_build(build: Any) -> tuple[str, dict[str, Any]]:
    """``"name"`` or ``("name", {overrides})`` -> ``(name, overrides)``."""
    if isinstance(build, str):
        return build, {}
    if isinstance(build, (tuple, list)) and len(build) == 2 and isinstance(build[0], str):
        if not isinstance(build[1], Mapping):
            raise ValidationError("build overrides must be a mapping")
        return build[0], dict(build[1])
    raise ValidationError("build must be 'name' or ('name', {overrides})")


def build_model(build: Any) -> Model:
    """A built-in model name, or an importable ``"module:function"``, plus keyword overrides."""
    name, overrides = normalise_build(build)
    if name in available():
        return load(name, **overrides)
    if ":" in name:
        module_name, func_name = name.split(":", 1)
        try:
            func = getattr(importlib.import_module(module_name), func_name)
        except (ImportError, AttributeError) as exc:
            raise ValidationError(f"cannot import model builder {name!r}: {exc}") from exc
        model = func(**overrides)
        if not isinstance(model, Model):
            raise ValidationError(f"{name!r} must return an dynecon Model")
        return model
    raise ValidationError(f"unknown model {name!r}; available: {available()} or 'module:function'")
