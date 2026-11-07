"""Adaptive random-walk Metropolis-Hastings with R-hat and effective sample size.

Economics note:
Bayesian estimation combines a prior with the likelihood to give the full posterior of the
structural parameters; MCMC draws summarise it and their diagnostics say whether the
sampler can be trusted.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ConvergenceWarning, ValidationError
from dynecon.estimation.results import MCMCResult
from dynecon.model.compiled import CompiledModel
from dynecon.utils.rng import as_generator
from dynecon.utils.validation import check_array, check_finite, check_int

__all__ = ["bayes_estimate", "ess", "metropolis", "split_rhat"]

Array = npt.NDArray[np.float64]


def check_start(x: npt.ArrayLike, name: str) -> Array:
    """The finite, non-empty 1-D starting point of a sampler."""
    x0 = np.atleast_1d(check_array(x, name))
    if x0.ndim != 1 or x0.size == 0:
        raise ValidationError(f"{name} must be a non-empty vector")
    return x0


def warn_if_flat(draws: Array, lps: Array, accept_rate: float) -> None:
    """Warn when chains moved freely on a perfectly flat log density (an improper target)."""
    moved = float(np.max(np.ptp(draws.reshape(-1, draws.shape[-1]), axis=0))) > 0.0
    if accept_rate > 0.999 and np.ptp(lps) == 0.0 and moved:
        warnings.warn(
            "the log density is constant over all draws: the target looks improper (flat) "
            "and the draws carry no information",
            ConvergenceWarning,
            stacklevel=3,
        )


def _logpost(f: Callable[[Array], float], x: Array) -> float:
    """``f(x)`` as a float that is never ``+inf`` (a chain would stick there forever)."""
    out = f(x)
    try:
        lp = float(out)
    except (TypeError, ValueError):
        raise ValidationError(f"the log posterior must return a real number, got {out!r}") from None
    if lp == np.inf:
        raise ValidationError("the log posterior returned +inf")
    return lp


def _unit_scale(chains: Array) -> Array:
    """Chains divided by their largest magnitude: R-hat and ESS ignore the scale, and the
    variances no longer overflow for draws near 1e154."""
    top = float(np.max(np.abs(chains)))
    return chains / top if top > 0.0 else chains


def split_rhat(chains: Array) -> float:
    """Split-chain potential scale reduction for draws ``(n_chains, n_draws)``."""
    chains = _unit_scale(chains)
    m, n = chains.shape
    half = n // 2
    seqs = np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)
    means, variances = seqs.mean(axis=1), seqs.var(axis=1, ddof=1)
    w = variances.mean()
    b = half * means.var(ddof=1)
    if w == 0.0:
        return 1.0
    var_plus = (half - 1) / half * w + b / half
    return float(np.sqrt(var_plus / w))


def ess(chains: Array) -> float:
    """Effective sample size (Geyer initial-positive-sequence) for draws ``(n_chains, n_draws)``."""
    chains = _unit_scale(chains)
    m, n = chains.shape
    x = chains - chains.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    chain_var = acov[:, 0] * n / (n - 1)
    w = chain_var.mean()
    b = n * chains.mean(axis=1).var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * w + b / n
    if var_plus == 0.0:
        return float(m * n)
    rho = 1.0 - (w - acov.mean(axis=0) * n / (n - 1)) / var_plus
    rho[0] = 1.0
    total = 0.0
    prev_pair = np.inf
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0.0:
            break
        pair = min(pair, prev_pair)
        total += pair
        prev_pair = pair
    tau = -1.0 + 2.0 * total
    return float(min(m * n / max(tau, 1e-12), m * n * np.log10(m * n)))


def metropolis(
    logpost: Callable[[Array], float],
    theta0: npt.ArrayLike,
    n_draws: int,
    *,
    n_chains: int = 4,
    burn_in: int = 1000,
    adapt_start: int = 200,
    init_scale: float = 0.5,
    rng: np.random.Generator | int | None = None,
    names: Sequence[str] | None = None,
) -> MCMCResult:
    """Random-walk Metropolis with Haario et al. covariance adaptation.

    Chains start at ``theta0`` plus ``N(0, init_scale^2)`` jitter; the proposal covariance is
    ``2.38^2 / d`` times the running sample covariance (plus a small ridge) after
    ``adapt_start`` iterations. ``burn_in`` draws are discarded.
    """
    x0 = check_start(theta0, "theta0")
    d = x0.size
    n_draws = check_int(n_draws, "n_draws", 10)
    n_chains = check_int(n_chains, "n_chains", 1)
    burn_in = check_int(burn_in, "burn_in", 0)
    adapt_start = check_int(adapt_start, "adapt_start", 0)
    init_scale = check_finite(init_scale, "init_scale", positive=True)
    gen = as_generator(rng)
    total = burn_in + n_draws
    out = np.empty((n_chains, n_draws, d))
    lps = np.empty((n_chains, n_draws))
    accepts = 0
    for c in range(n_chains):
        x = x0 + init_scale * gen.standard_normal(d)
        lp = _logpost(logpost, x)
        tries = 0
        while not np.isfinite(lp) and tries < 100:
            x = x0 + init_scale * gen.standard_normal(d) * 0.1 ** (tries // 10)
            lp, tries = _logpost(logpost, x), tries + 1
        if not np.isfinite(lp):
            raise ValidationError("could not find a starting point with finite posterior")
        mean, m2, count = x.copy(), np.zeros((d, d)), 1
        prop_chol = np.linalg.cholesky(0.1**2 * np.eye(d))
        for t in range(total):
            cand = x + prop_chol @ gen.standard_normal(d)
            lp_c = _logpost(logpost, cand)
            if np.log(gen.random()) < lp_c - lp:
                x, lp = cand, lp_c
                if t >= burn_in:
                    accepts += 1
            count += 1
            delta = x - mean
            mean += delta / count
            m2 += np.outer(delta, x - mean)
            if t + 1 >= adapt_start:
                cov = m2 / (count - 1)
                prop_chol = np.linalg.cholesky(2.38**2 / d * cov + 1e-8 * np.eye(d))
            if t >= burn_in:
                out[c, t - burn_in] = x
                lps[c, t - burn_in] = lp
    labels = list(names) if names is not None else [f"theta[{i}]" for i in range(d)]
    rhat = np.array([split_rhat(out[:, :, i]) for i in range(d)])
    ess_v = np.array([ess(out[:, :, i]) for i in range(d)])
    accept_rate = accepts / (n_chains * n_draws)
    warn_if_flat(out, lps, accept_rate)
    return MCMCResult(out, labels, accept_rate, rhat, ess_v, lps)


def bayes_estimate(
    model: CompiledModel,
    free_params: Sequence[str],
    loglik: Callable[[CompiledModel], float],
    n_draws: int,
    *,
    n_chains: int = 4,
    burn_in: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> MCMCResult:
    """Posterior draws of ``free_params`` from ``prior x likelihood`` (draws in parameter space).

    Sampling is done on the unconstrained scale given by each parameter's transform; the
    log-Jacobian is added so the target is the posterior of the original parameters.
    Parameters without a prior get a flat prior on their bounds.
    """
    names = list(free_params)
    ps = model.parameters

    def logpost(u: Array) -> float:
        theta = ps.from_unconstrained(u, names)
        lp = ps.log_jacobian(u, names)
        for n, v in zip(names, theta, strict=True):
            prior = ps[n].prior
            if prior is not None:
                lp += prior.logpdf(float(v))
        if not np.isfinite(lp):
            return float("-inf")
        try:
            ll = float(loglik(model.with_params(**dict(zip(names, theta, strict=True)))))
        except (ValidationError, ValueError, ArithmeticError):
            return float("-inf")
        return lp + ll if np.isfinite(ll) else float("-inf")

    u0 = ps.to_unconstrained(ps.vector(names), names)
    res = metropolis(logpost, u0, n_draws, n_chains=n_chains, burn_in=burn_in, rng=rng, names=names)
    theta_draws = np.empty_like(res.draws)
    for c in range(res.draws.shape[0]):
        for t in range(res.draws.shape[1]):
            theta_draws[c, t] = ps.from_unconstrained(res.draws[c, t], names)
    rhat = np.array([split_rhat(theta_draws[:, :, i]) for i in range(len(names))])
    ess_v = np.array([ess(theta_draws[:, :, i]) for i in range(len(names))])
    return MCMCResult(theta_draws, names, res.accept_rate, rhat, ess_v, res.log_posterior)
