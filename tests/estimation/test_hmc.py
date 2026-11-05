import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.estimation import hmc, metropolis


def _gauss(cov: np.ndarray):
    inv = np.linalg.inv(cov)
    return lambda x: (float(-0.5 * x @ inv @ x), -inv @ x)


def test_recovers_mean_and_covariance_of_a_correlated_gaussian() -> None:
    cov = np.array([[1.0, 0.8, 0.0], [0.8, 2.0, 0.3], [0.0, 0.3, 0.5]])
    res = hmc(_gauss(cov), np.zeros(3), 4000, n_chains=2, n_warmup=400, rng=5)
    np.testing.assert_allclose(res.mean, 0.0, atol=0.08)
    np.testing.assert_allclose(np.cov(res.pooled.T), cov, rtol=0.1, atol=0.05)
    assert res.divergences == 0 and np.all(res.rhat < 1.02) and np.all(res.ess > 500)
    assert 0.6 < res.accept_rate < 0.99


def test_beats_random_walk_per_target_evaluation() -> None:
    cov = np.diag(np.linspace(0.5, 2.0, 6))
    calls = {"hmc": 0, "rw": 0}
    f = _gauss(cov)

    def counted(x: np.ndarray):
        calls["hmc"] += 1
        return f(x)

    def lp(x: np.ndarray) -> float:
        calls["rw"] += 1
        return f(x)[0]

    h = hmc(counted, np.zeros(6), 1000, n_chains=2, n_warmup=300, rng=1)
    m = metropolis(lp, np.zeros(6), 3000, n_chains=2, burn_in=500, rng=1)
    assert h.ess.mean() / calls["hmc"] > 2 * m.ess.mean() / calls["rw"]


def test_failed_target_regions_are_rejected_and_counted() -> None:
    def half_line(x: np.ndarray):  # density zero (non-finite) for x < -1.5: trajectories can hit it
        if x[0] < -1.5:
            return float("-inf"), np.zeros(1)
        return float(-0.5 * x @ x), -x

    res = hmc(half_line, np.zeros(1), 500, n_chains=2, n_warmup=200, rng=2)
    assert np.all(res.pooled[:, 0] >= -1.5)
    assert res.divergences >= 0


def test_bad_arguments_raise() -> None:
    f = _gauss(np.eye(2))
    with pytest.raises(ValidationError):
        hmc(f, np.zeros(2), 5)
    with pytest.raises(ValidationError):
        hmc(f, np.zeros(2), 100, n_warmup=5)
    with pytest.raises(ValidationError):
        hmc(lambda x: (float("-inf"), np.zeros(2)), np.zeros(2), 100)
