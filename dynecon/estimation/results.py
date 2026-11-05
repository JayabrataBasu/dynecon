"""Immutable result objects of the estimators.

Economics note:
An estimate is only useful with its uncertainty and its diagnostics; results therefore
bundle parameter estimates, standard errors, fit statistics and the configuration used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["MCMCResult", "MLEResult", "SMMResult"]

Array = npt.NDArray[np.float64]


def _table(names: list[str], cols: dict[str, Array]) -> str:
    head = f"{'parameter':<14s}" + "".join(f"{k:>13s}" for k in cols)
    rows = [
        f"{n:<14s}" + "".join(f"{v[i]:13.5g}" for v in cols.values()) for i, n in enumerate(names)
    ]
    return "\n".join([head, *rows])


def _no_loss(params: dict[str, float]) -> float:
    del params
    return float("nan")


@dataclass(frozen=True)
class SMMResult:
    """Simulated-method-of-moments estimate with inference and diagnostics."""

    names: list[str]
    estimates: Array
    std_errors: Array
    loss: float
    weights: Array
    omega: Array
    jacobian: Array
    model_moments: Array
    data_moments: Array
    n_obs: int
    n_sim: float
    j_stat: float | None
    j_pvalue: float | None
    df: int
    converged: bool
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    loss_fn: Callable[[dict[str, float]], float] = field(repr=False, default=_no_loss)

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.estimates), strict=True))

    @property
    def t_stats(self) -> Array:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(self.estimates / self.std_errors, dtype=np.float64)

    def to_dataframe(self) -> Any:
        """Estimates table as a ``pandas.DataFrame`` (needs pandas)."""
        import pandas as pd

        return pd.DataFrame(
            {"estimate": self.estimates, "std_error": self.std_errors, "t_stat": self.t_stats},
            index=self.names,
        )

    def summary(self) -> str:
        """Text report: estimates, standard errors, J-test and diagnostics."""
        cols = {"estimate": self.estimates, "std.err": self.std_errors, "t": self.t_stats}
        lines = ["SMM estimates", _table(self.names, cols)]
        lines.append(
            f"loss = {self.loss:.4g}, moments = {self.data_moments.size}, N = {self.n_obs}"
        )
        if self.j_stat is not None:
            lines.append(f"J = {self.j_stat:.4g} (df {self.df}), p-value = {self.j_pvalue:.4g}")
        lines.append(f"failed evaluations: {self.diagnostics.get('failed_evals', 0)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class MLEResult:
    """Maximum-likelihood estimate with Hessian-based standard errors."""

    estimates: Array
    std_errors: Array
    loglik: float
    hessian: Array
    covariance: Array
    converged: bool
    n_iter: int

    def summary(self) -> str:
        names = [f"theta[{i}]" for i in range(self.estimates.size)]
        cols = {"estimate": self.estimates, "std.err": self.std_errors}
        return "\n".join(["MLE estimates", _table(names, cols), f"loglik = {self.loglik:.6g}"])


@dataclass(frozen=True)
class MCMCResult:
    """Metropolis-Hastings draws ``(n_chains, n_draws, n_params)`` with convergence diagnostics."""

    draws: Array
    names: list[str]
    accept_rate: float
    rhat: Array
    ess: Array
    log_posterior: Array = field(repr=False, default_factory=lambda: np.zeros(0))
    divergences: int = 0  # HMC only: trajectories rejected for energy blow-up or a failed target

    @property
    def pooled(self) -> Array:
        """All chains stacked: ``(n_chains * n_draws, n_params)``."""
        return self.draws.reshape(-1, self.draws.shape[-1])

    @property
    def mean(self) -> Array:
        return np.asarray(self.pooled.mean(axis=0), dtype=np.float64)

    @property
    def std(self) -> Array:
        return np.asarray(self.pooled.std(axis=0, ddof=1), dtype=np.float64)

    def summary(self) -> str:
        """Posterior mean, std, R-hat and effective sample size per parameter."""
        cols = {"mean": self.mean, "std": self.std, "rhat": self.rhat, "ess": self.ess}
        return "\n".join(
            ["Posterior summary", _table(self.names, cols), f"acceptance = {self.accept_rate:.3f}"]
        )
