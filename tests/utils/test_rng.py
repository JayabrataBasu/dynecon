import numpy as np
import pytest

from dynecon.utils.rng import as_generator, spawn


def test_as_generator_reproducible() -> None:
    np.testing.assert_array_equal(as_generator(3).random(5), as_generator(3).random(5))


def test_as_generator_passthrough() -> None:
    g = np.random.default_rng(1)
    assert as_generator(g) is g


def test_spawn_independent_and_reproducible() -> None:
    a = [g.standard_normal(100_000) for g in spawn(7, 3)]
    b = [g.standard_normal(100_000) for g in spawn(7, 3)]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(np.corrcoef(a[i], a[j])[0, 1]) < 0.01


def test_spawn_negative_raises() -> None:
    with pytest.raises(ValueError):
        spawn(0, -1)
