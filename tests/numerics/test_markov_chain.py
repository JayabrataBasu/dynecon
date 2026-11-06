import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.numerics import MarkovChain


def _sets(classes):
    return sorted(tuple(c.tolist()) for c in classes)


def test_structure_of_a_reducible_chain() -> None:
    # 0 <-> 1 closed, 2 -> {0,1}, 3 absorbing
    P = np.array([[0.5, 0.5, 0, 0], [0.3, 0.7, 0, 0], [0.2, 0.2, 0.6, 0], [0, 0, 0, 1.0]])
    mc = MarkovChain(P)
    assert _sets(mc.communication_classes) == [(0, 1), (2,), (3,)]
    assert _sets(mc.recurrent_classes) == [(0, 1), (3,)]
    assert not mc.is_irreducible and mc.is_aperiodic
    pis = mc.stationary_distributions
    assert pis.shape == (2, 4)
    np.testing.assert_allclose(pis[0], [0.375, 0.625, 0, 0])
    np.testing.assert_allclose(pis[1], [0, 0, 0, 1])
    np.testing.assert_allclose(pis @ P, pis, atol=1e-14)


def test_period_and_aperiodicity() -> None:
    cycle3 = np.roll(np.eye(3), 1, axis=1)
    assert MarkovChain(cycle3).period == 3 and not MarkovChain(cycle3).is_aperiodic
    lazy = 0.5 * np.eye(3) + 0.5 * cycle3
    assert MarkovChain(lazy).period == 1 and MarkovChain(lazy).is_aperiodic
    two = np.array([[0, 1.0], [1.0, 0]])
    assert MarkovChain(two).period == 2
    with pytest.raises(ValidationError):
        MarkovChain(np.eye(2)).period


def test_irreducible_chain_matches_stationary_distribution_and_simulates() -> None:
    from dynecon.numerics import stationary_distribution

    P = np.array([[0.9, 0.1, 0.0], [0.2, 0.5, 0.3], [0.1, 0.4, 0.5]])
    mc = MarkovChain(P, state_values=np.array([10.0, 20.0, 30.0]))
    assert mc.is_irreducible
    np.testing.assert_allclose(mc.stationary_distributions[0], stationary_distribution(P))
    path = mc.simulate(20_000, rng=1)
    assert set(np.unique(path)) <= {10.0, 20.0, 30.0}
    freq = np.array([(path == v).mean() for v in (10.0, 20.0, 30.0)])
    np.testing.assert_allclose(freq, stationary_distribution(P), atol=0.02)
    with pytest.raises(ValidationError):
        MarkovChain(np.eye(2)).simulate(5)  # reducible: needs init
    assert list(MarkovChain(np.eye(2)).simulate(3, init=1)) == [1.0, 1.0, 1.0]
    with pytest.raises(ValidationError):
        MarkovChain(P, state_values=np.ones(2))
