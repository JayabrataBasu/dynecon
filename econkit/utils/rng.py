"""Random-number contract: every stochastic function takes ``rng`` and calls ``as_generator``."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["as_generator", "spawn"]


def as_generator(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    """Return a PCG64 ``Generator`` from a generator, an integer seed, or ``None``.

    Economics note:
    Reproducible shocks are what make simulated moments comparable across
    parameter values (common random numbers) and runs auditable.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.Generator(np.random.PCG64(rng))


def spawn(rng: np.random.Generator | int | None, n: int) -> list[np.random.Generator]:
    """Return ``n`` statistically independent child generators.

    Economics note:
    Independent streams give each simulated agent (or replication) its own shock
    history while keeping the whole panel reproducible from one seed.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    gen = as_generator(rng)
    seed_seq: Any = gen.bit_generator.seed_seq
    return [np.random.Generator(np.random.PCG64(s)) for s in seed_seq.spawn(n)]
