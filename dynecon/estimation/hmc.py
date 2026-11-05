"""Hamiltonian Monte Carlo with step-size and diagonal-mass adaptation.

Economics note:
Random-walk samplers need thousands of model solves per effective draw. When the posterior
(or quasi-posterior) has an exact gradient, Hamiltonian dynamics move far across the
distribution per step, so far fewer draws, and therefore far fewer model solves, are needed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.estimation.bayes import ess, split_rhat
from dynecon.estimation.results import MCMCResult
from dynecon.utils.rng import as_generator

__all__ = ["hmc"]

Array = npt.NDArray[np.float64]
LogpGrad = Callable[[Array], tuple[float, Array]]
MAX_ENERGY_ERROR = 1000.0  # a trajectory whose energy drifts this far is called divergent


def _leapfrog(
    f: LogpGrad, u: Array, g: Array, p: Array, inv_mass: Array, eps: float, n: int
) -> tuple[Array, float, Array, Array] | None:
    """``n`` leapfrog steps; returns ``(u, lp, grad, p)`` or ``None`` if the target failed."""
    p = p + 0.5 * eps * g
    for i in range(n):
        u = u + eps * (inv_mass @ p)
        lp, g = f(u)
        if not np.isfinite(lp) or not np.all(np.isfinite(g)):
            return None
        p = p + (eps if i < n - 1 else 0.5 * eps) * g
    return u, lp, g, p


def _transition(
    f: LogpGrad,
    u: Array,
    lp: float,
    g: Array,
    eps: float,
    inv_mass: Array,
    n_steps: int,
    gen: np.random.Generator,
) -> tuple[Array, float, Array, float, bool]:
    """One HMC transition. Returns ``(u, lp, grad, accept_prob, divergent)``."""
    p0 = np.linalg.cholesky(np.linalg.inv(inv_mass)) @ gen.standard_normal(u.size)
    h0 = -lp + 0.5 * p0 @ inv_mass @ p0
    out = _leapfrog(f, u, g, p0, inv_mass, eps, n_steps)
    if out is None:
        return u, lp, g, 0.0, True
    u1, lp1, g1, p1 = out
    err = float(-lp1 + 0.5 * p1 @ inv_mass @ p1 - h0)
    accept = min(1.0, float(np.exp(-err))) if np.isfinite(err) else 0.0
    divergent = bool(not np.isfinite(err) or err > MAX_ENERGY_ERROR)
    if not divergent and gen.random() < accept:
        return u1, lp1, g1, accept, False
    return u, lp, g, accept, divergent


def _steps(eps: float, traj: float, max_leapfrog: int, gen: np.random.Generator) -> int:
    """Jittered number of leapfrog steps for a trajectory of length about ``traj``."""
    return int(np.clip(np.ceil(traj * gen.uniform(0.8, 1.2) / eps), 1, max_leapfrog))


def _initial_eps(
    f: LogpGrad,
    u: Array,
    lp: float,
    g: Array,
    inv_mass: Array,
    traj: float,
    max_leapfrog: int,
    gen: np.random.Generator,
) -> float:
    """Double or halve ``eps`` until one transition's acceptance crosses 1/2."""
    eps = 0.1
    acc = _transition(f, u, lp, g, eps, inv_mass, _steps(eps, traj, max_leapfrog, gen), gen)[3]
    direction = 1.0 if acc > 0.5 else -1.0
    for _ in range(30):
        eps *= 2.0**direction
        acc = _transition(f, u, lp, g, eps, inv_mass, _steps(eps, traj, max_leapfrog, gen), gen)[3]
        if (direction > 0) != (acc > 0.5):
            break
    return float(eps)


def _warmup(  # noqa: C901
    f: LogpGrad,
    u: Array,
    lp: float,
    g: Array,
    n_warmup: int,
    target: float,
    traj: float,
    max_leapfrog: int,
    gen: np.random.Generator,
    inv_mass0: Array | None = None,
) -> tuple[Array, float, Array, float, Array]:
    """Dual-averaging step size (Hoffman & Gelman 2014) with a diagonal mass from the first half."""
    d = u.size
    inv_mass = np.eye(d) if inv_mass0 is None else inv_mass0
    half = n_warmup // 2
    seen: list[Array] = []
    for stage, length in enumerate((half, n_warmup - half)):
        eps = _initial_eps(f, u, lp, g, inv_mass, traj, max_leapfrog, gen)
        mu, h_bar, log_bar = np.log(10.0 * eps), 0.0, 0.0
        for t in range(1, length + 1):
            n = _steps(eps, traj, max_leapfrog, gen)
            u, lp, g, acc, _ = _transition(f, u, lp, g, eps, inv_mass, n, gen)
            h_bar = (1 - 1 / (t + 10)) * h_bar + (target - acc) / (t + 10)
            log_eps = mu - np.sqrt(t) / 0.05 * h_bar
            w = t**-0.75
            log_bar = w * log_eps + (1 - w) * log_bar
            eps = float(np.exp(log_eps))
            if stage == 0:
                seen.append(u.copy())
        if stage == 0 and len(seen) > 10 and inv_mass0 is None:
            var = np.var(np.array(seen), axis=0, ddof=1)
            k = len(seen)  # shrink towards 1 as in Stan
            inv_mass = np.diag(np.maximum((k / (k + 5)) * var + 1e-3 * 5 / (k + 5), 1e-12))
    return u, lp, g, float(np.exp(log_bar)), inv_mass


def hmc(
    logp_grad: LogpGrad,
    u0: npt.ArrayLike,
    n_draws: int,
    *,
    n_chains: int = 4,
    n_warmup: int = 500,
    target_accept: float = 0.8,
    trajectory: float = 1.5,
    max_leapfrog: int = 64,
    init_scale: float = 0.1,
    rng: np.random.Generator | int | None = None,
    names: Sequence[str] | None = None,
    inv_mass: npt.ArrayLike | None = None,
) -> MCMCResult:
    """Sample ``exp(logp)`` with Hamiltonian Monte Carlo; ``logp_grad(u)`` returns ``(logp, grad)``.

    Each chain starts at ``u0`` plus ``N(0, init_scale^2)`` jitter. The first half of
    ``n_warmup`` adapts the step size by dual averaging towards ``target_accept`` and collects
    draws for a diagonal mass matrix; the second half re-adapts the step size under that mass.
    Trajectories have length about ``trajectory`` (in mass-scaled units, jittered 20%) and at
    most ``max_leapfrog`` steps. Divergent trajectories are rejected and counted. Returns
    :class:`MCMCResult` (``accept_rate`` is the mean acceptance probability).

    ``inv_mass`` (a vector or a symmetric positive-definite matrix, e.g. the inverse Hessian at a
    posterior mode) replaces the adapted diagonal mass: with a dense mass the step size can be
    large even when parameters are strongly correlated.
    """
    x0 = np.asarray(u0, dtype=np.float64)
    d = x0.size
    if n_draws < 10 or n_chains < 1 or n_warmup < 20 or not 0.0 < target_accept < 1.0:
        raise ValidationError("need n_draws >= 10, n_chains >= 1, n_warmup >= 20, 0<target<1")
    gen = as_generator(rng)
    fixed = None if inv_mass is None else np.asarray(inv_mass, dtype=np.float64)
    if fixed is not None and fixed.ndim == 1:
        fixed = np.diag(fixed)
    if fixed is not None and fixed.shape != (d, d):
        raise ValidationError(f"inv_mass must be a vector or matrix of size {d}")
    out = np.empty((n_chains, n_draws, d))
    lps = np.empty((n_chains, n_draws))
    acc_sum, divergences = 0.0, 0
    for c in range(n_chains):
        for attempt in range(100):
            u = x0 + init_scale * 0.5 ** (attempt // 10) * gen.standard_normal(d)
            lp, g = logp_grad(u)
            if np.isfinite(lp) and np.all(np.isfinite(g)):
                break
        else:
            raise ValidationError("could not find a starting point with a finite log density")
        u, lp, g, eps, inv_m = _warmup(
            logp_grad, u, lp, g, n_warmup, target_accept, trajectory, max_leapfrog, gen, fixed
        )
        for t in range(n_draws):
            n = _steps(eps, trajectory, max_leapfrog, gen)
            u, lp, g, acc, bad = _transition(logp_grad, u, lp, g, eps, inv_m, n, gen)
            acc_sum += acc
            divergences += int(bad)
            out[c, t], lps[c, t] = u, lp
    labels = list(names) if names is not None else [f"theta[{i}]" for i in range(d)]
    rhat = np.array([split_rhat(out[:, :, i]) for i in range(d)])
    ess_v = np.array([ess(out[:, :, i]) for i in range(d)])
    return MCMCResult(out, labels, acc_sum / (n_chains * n_draws), rhat, ess_v, lps, divergences)
