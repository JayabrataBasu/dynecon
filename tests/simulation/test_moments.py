import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.simulation import moments as m


def ar1(rho: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def test_basic_moments() -> None:
    x = np.arange(10.0)
    assert m.sample_mean(x) == 4.5
    assert abs(m.sample_std(x) - np.std(x)) < 1e-15
    assert m.percentile(x, 50) == 4.5
    assert abs(m.correlation(x, 2 * x + 1) - 1.0) < 1e-12
    assert abs(m.regression_slope(3 * x + 2, x) - 3.0) < 1e-12
    assert m.correlation(np.ones(5), np.arange(5.0)) == 0.0


def test_autocorrelation_within_bootstrap_standard_errors() -> None:
    rho = 0.9
    x = ar1(rho, 200_000, seed=1)
    ac = m.autocorrelation(x, 1)
    se = m.block_bootstrap_se(
        x, lambda s: m.autocorrelation(s, 1), block_len=500, n_boot=100, rng=0
    )
    assert abs(ac - rho) < 4 * se


def test_panel_autocorrelation_pairs_within_agent() -> None:
    panel = np.column_stack([np.arange(50.0), np.arange(50.0) * -1 + 100])
    assert m.autocorrelation(panel, 1) > 0.99


def test_newey_west_matches_iid_variance_of_mean() -> None:
    x = np.random.default_rng(0).standard_normal(50_000)
    assert abs(m.newey_west_variance(x) * x.size - 1.0) < 0.05
    assert m.newey_west_variance(x, lags=0) > 0


def test_validation_errors() -> None:
    with pytest.raises(ValidationError):
        m.autocorrelation(np.arange(3.0), 5)
    with pytest.raises(ValidationError):
        m.autocorrelation(np.arange(10.0), 0)
    with pytest.raises(ValidationError):
        m.correlation(np.ones(3), np.ones(4))
    with pytest.raises(ValidationError):
        m.regression_slope(np.ones(3), np.ones(3))
    with pytest.raises(ValidationError):
        m.regression_slope(np.ones(3), np.ones(4))
    with pytest.raises(ValidationError):
        m.block_bootstrap_se(np.ones((3, 3)), np.mean, 2)
    with pytest.raises(ValidationError):
        m.block_bootstrap_se(np.ones(5), np.mean, 0)
    with pytest.raises(ValidationError):
        m.block_bootstrap_se(np.ones(5), 3, 2)  # type: ignore[arg-type]
