"""JAX global setup: enable 64-bit precision on first import of any JAX backend module.

Every tolerance in Dynecon assumes ``float64``, so there is deliberately no opt-out.
Importing this module has the global side effect ``jax_enable_x64 = True``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

X64_ENABLED: bool = bool(jax.config.read("jax_enable_x64"))
