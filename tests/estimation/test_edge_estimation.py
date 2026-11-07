"""Degenerate estimation inputs give a clear ValidationError."""

import warnings

import numpy as np
import pytest

from dynecon.core.errors import ConvergenceWarning, ValidationError
from dynecon.estimation import hmc, kalman_filter, metropolis


def test_non_finite_state_space_is_rejected() -> None:
    one, y = np.eye(1), np.ones((5, 1))
    with pytest.raises(ValidationError):
        kalman_filter(y, one, one, np.array([[np.nan]]), one, np.zeros(1), one)
    with pytest.raises(ValidationError):
        kalman_filter(y, one, one, one, one, np.array([np.inf]), one)


def test_empty_starting_point_is_rejected() -> None:
    with pytest.raises(ValidationError):
        metropolis(lambda x: 0.0, np.zeros(0), 100)
    with pytest.raises(ValidationError):
        hmc(lambda x: (0.0, np.zeros(0)), np.zeros(0), 100)


def test_bootstrap_rejects_non_finite_data() -> None:
    from dynecon.experiment import bootstrap_ci

    with pytest.raises(ValidationError):
        bootstrap_ci(np.array([1.0, np.nan, 3.0]), np.mean, n_boot=10)


def test_hmc_validates_the_target_output() -> None:
    def short(x):
        return -0.5 * x @ x, np.zeros(3)  # gradient of the wrong length for a 2-vector

    with pytest.raises(ValidationError, match="shape"):
        hmc(short, np.zeros(2), 50, n_chains=1, n_warmup=30)
    with pytest.raises(ValidationError, match="logp_grad"):
        hmc(lambda x: -0.5 * x @ x, np.zeros(2), 50, n_chains=1, n_warmup=30)
    with pytest.raises(ValidationError, match="logp_grad"):
        hmc(lambda x: ("a", np.zeros(2)), np.zeros(2), 50, n_chains=1, n_warmup=30)


@pytest.mark.parametrize("sampler", [metropolis, hmc])
@pytest.mark.parametrize(
    "kw",
    [
        {"n_draws": 2.5},
        {"n_draws": np.nan},
        {"n_draws": 5},
        {"n_chains": 0},
        {"n_chains": 1.5},
        {"init_scale": np.nan},
        {"init_scale": -1.0},
    ],
)
def test_samplers_reject_bad_options(sampler, kw) -> None:
    target = (lambda x: (-0.5 * x @ x, -x)) if sampler is hmc else (lambda x: -0.5 * x @ x)
    opts = {"n_draws": 50, "n_chains": 1, **kw}
    with pytest.raises(ValidationError):
        sampler(target, np.zeros(2), **opts)


@pytest.mark.parametrize("start", [[np.nan, 0.0], [np.inf], [[0.0, 1.0], [2.0, 3.0]], ["a"]])
def test_samplers_reject_bad_start(start) -> None:
    with pytest.raises(ValidationError):
        metropolis(lambda x: 0.0, start, 50)
    with pytest.raises(ValidationError):
        hmc(lambda x: (0.0, np.zeros(2)), start, 50)


def test_flat_target_warns_and_proper_targets_do_not() -> None:
    with pytest.warns(ConvergenceWarning, match="constant"):
        metropolis(lambda x: 0.0, np.zeros(2), 200, n_chains=2, burn_in=50)
    with pytest.warns(ConvergenceWarning, match="constant"):
        hmc(lambda x: (0.0, np.zeros(2)), np.zeros(2), 50, n_chains=1, n_warmup=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metropolis(lambda x: -0.5 * x @ x, np.zeros(2), 200, n_chains=2, burn_in=100, rng=1)
        hmc(lambda x: (-0.5 * x @ x, -x), np.zeros(2), 50, n_chains=1, n_warmup=100, rng=1)
        # a bounded uniform target is proper: proposals outside the box are rejected
        box = lambda x: 0.0 if np.all(np.abs(x) < 1.0) else -np.inf  # noqa: E731
        metropolis(box, np.zeros(2), 400, n_chains=2, burn_in=100, rng=1)


def test_metropolis_rejects_non_scalar_and_infinite_log_posterior() -> None:
    with pytest.raises(ValidationError, match="real number"):
        metropolis(lambda x: x, np.zeros(2), 50)
    with pytest.raises(ValidationError, match="real number"):
        metropolis(lambda x: "a", np.zeros(2), 50)
    with pytest.raises(ValidationError, match=r"\+inf"):
        metropolis(
            lambda x: np.inf if x[0] > 0.5 else -0.5 * x @ x,
            np.zeros(2),
            400,
            n_chains=1,
            burn_in=0,
        )


def test_diagnostics_do_not_overflow_for_huge_draws() -> None:
    from dynecon.estimation.bayes import ess, split_rhat

    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, 200))
    for scale in (1.0, 1e100, 4e153):
        assert abs(split_rhat(base * scale) - split_rhat(base)) < 1e-9
        assert abs(ess(base * scale) - ess(base)) < 1e-6
    res = metropolis(
        lambda x: -0.5 * x @ x, np.zeros(2), 10, n_chains=5, burn_in=3, init_scale=4e153, rng=0
    )
    assert np.all(np.isfinite(res.rhat)) and np.all(np.isfinite(res.ess))
