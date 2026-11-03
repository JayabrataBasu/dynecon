"""Built-in canonical models behind a uniform ``available()`` / ``load()`` interface."""

from dynecon.models import (  # noqa: F401
    aiyagari,
    brock_mirman,
    consumption_savings,
    huggett,
    lucas_tree,
    mccall,
    nk3,
    olg2,
    rbc,
)
from dynecon.models.registry import available, load, register

__all__ = ["available", "load", "register"]
