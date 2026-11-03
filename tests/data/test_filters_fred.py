"""Filters against exact properties; FRED parsing from a bundled sample (no network)."""

import numpy as np
import pytest

from dynecon.core.errors import DyneconError, ValidationError
from dynecon.data import baxter_king, fetch_fred, hamilton_filter, hp_filter, parse_fred_csv

SAMPLE = """DATE,GDPC1
2000-01-01,100.0
2000-04-01,101.5
2000-07-01,.
2000-10-01,103.2
"""


def test_hp_linear_trend_has_zero_cycle_and_identity() -> None:
    t = np.arange(100.0)
    trend, cycle = hp_filter(2.0 + 0.5 * t)
    np.testing.assert_allclose(cycle, 0.0, atol=1e-8)
    y = np.random.default_rng(0).standard_normal(200)
    trend, cycle = hp_filter(y)
    np.testing.assert_allclose(trend + cycle, y, atol=1e-12)
    assert np.var(cycle) < np.var(y)
    with pytest.raises(ValidationError):
        hp_filter([1.0, 2.0, 3.0])
    with pytest.raises(ValidationError):
        hp_filter(y, lamb=-1.0)
    with pytest.raises(ValidationError):
        hp_filter(np.array([1.0, np.nan, 3.0, 4.0, 5.0]))


def test_baxter_king_weights_and_passband() -> None:
    n = 400
    t = np.arange(n)
    out = baxter_king(np.sin(2 * np.pi * t / 16))
    assert out.shape == (n - 24,)
    gain = np.std(out) / np.std(np.sin(2 * np.pi * t / 16))
    assert abs(gain - 1.0) < 0.05
    np.testing.assert_allclose(baxter_king(np.full(n, 5.0)), 0.0, atol=1e-8)
    np.testing.assert_allclose(baxter_king(0.3 * t + 2.0), 0.0, atol=1e-8)
    # weights sum to zero and are symmetric: check through the response to an impulse
    impulse = np.zeros(61)
    impulse[30] = 1.0
    w = baxter_king(impulse)
    assert abs(w.sum()) < 1e-12 and np.allclose(w, w[::-1])
    with pytest.raises(ValidationError):
        baxter_king(np.ones(10))
    with pytest.raises(ValidationError):
        baxter_king(np.ones(100), low=10, high=5)


def test_hamilton_polynomial_and_random_walk() -> None:
    t = np.arange(120.0)
    trend, cycle = hamilton_filter(1.0 + 0.3 * t + 0.01 * t**2 + 0.001 * t**3, h=8, p=4)
    np.testing.assert_allclose(cycle, 0.0, atol=1e-6)
    rw = np.cumsum(np.random.default_rng(1).standard_normal(400))
    trend, cycle = hamilton_filter(rw, h=8, p=4)
    assert len(cycle) == 400 - 8 - 4 + 1
    # the residual is the h-step forecast error: uncorrelated with the regressors
    X = np.column_stack([np.ones(len(cycle))] + [rw[3 - k : 3 - k + len(cycle)] for k in range(4)])
    np.testing.assert_allclose(X.T @ np.asarray(cycle), 0.0, atol=1e-6)
    with pytest.raises(ValidationError):
        hamilton_filter(np.ones(5))


def test_pandas_index_preserved() -> None:
    pd = pytest.importorskip("pandas")
    s = pd.Series(
        np.random.default_rng(2).standard_normal(100),
        index=pd.date_range("2000", periods=100, freq="QS"),
    )
    trend, cycle = hp_filter(s)
    assert trend.index.equals(s.index) and isinstance(cycle, pd.Series)
    bk = baxter_king(s)
    assert bk.index.equals(s.index[12:-12])
    ht, hc = hamilton_filter(s)
    assert hc.index.equals(s.index[11:])


def test_parse_fred_csv_handles_missing_values() -> None:
    pytest.importorskip("pandas")
    s = parse_fred_csv(SAMPLE)
    assert s.name == "GDPC1" and len(s) == 4
    assert np.isnan(s.iloc[2]) and s.iloc[3] == pytest.approx(103.2)
    assert s.index.is_monotonic_increasing and s.index.name == "date"
    with pytest.raises(ValidationError):
        parse_fred_csv("a,b,c\n1,2,3\n")


def test_fetch_uses_cache_and_validates_without_network(tmp_path) -> None:
    pytest.importorskip("pandas")
    (tmp_path / "GDPC1_None_None.csv").write_text(SAMPLE, encoding="utf-8")
    assert len(fetch_fred("GDPC1", cache_dir=tmp_path)) == 4
    with pytest.raises(ValidationError):
        fetch_fred("bad id!")


def test_fetch_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    import urllib.error
    import urllib.request

    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(DyneconError, match="could not download"):
        fetch_fred("GDPC1")


@pytest.mark.network
def test_live_fred_download() -> None:  # pragma: no cover - never run in CI
    pytest.skip("network tests are skipped by default")
