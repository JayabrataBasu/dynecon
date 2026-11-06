"""HMC on the Smets-Wouters posterior with the exact gradient (see real_sw_grad.py).

Usage: python scripts/real_sw_hmc_exact.py DIR OUT.npz [n_draws] [n_chains]
One chain per process; logit-scaled parameters; mass matrix from the Hessian at the mode.
"""

from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from real_sw_estimate import SW
from real_sw_grad import loglik_and_grad
from scipy.optimize import minimize

from dynecon.estimation.bayes import ess, split_rhat
from dynecon.estimation.hmc import hmc

warnings.filterwarnings("ignore", category=RuntimeWarning)
_sw: SW | None = None
_lb = _ub = None


def _init(folder: str) -> None:
    global _sw, _lb, _ub
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _sw = SW(folder)
    _lb = np.array([e[2] for e in _sw.est])
    _ub = np.array([e[3] for e in _sw.est])


def _x(u):
    return _lb + (_ub - _lb) / (1 + np.exp(-u))


def logp_grad(u):
    """Log posterior + log-Jacobian at logit-scale ``u`` with its exact gradient."""
    s = 1 / (1 + np.exp(-u))
    x = _lb + (_ub - _lb) * s
    lpr = _sw.logprior(x)
    if not np.isfinite(lpr):
        return float("-inf"), np.zeros_like(u)
    try:
        ll, g = loglik_and_grad(_sw, x)
    except Exception:  # noqa: BLE001 - indeterminate or singular: density zero
        return float("-inf"), np.zeros_like(u)
    if not np.isfinite(ll) or not np.all(np.isfinite(g)):
        return float("-inf"), np.zeros_like(u)
    gp = np.empty_like(x)  # prior gradient by central differences (scalar functions)
    for i in range(len(x)):
        h = 1e-6 * max(1.0, abs(x[i]))
        up, dn = x.copy(), x.copy()
        up[i] += h
        dn[i] -= h
        gp[i] = (_sw.logprior(up) - _sw.logprior(dn)) / (2 * h)
    dx = (_ub - _lb) * s * (1 - s)
    lj = np.sum(np.log(_ub - _lb) + np.log(s) + np.log1p(-s))
    return float(ll + lpr + lj), (g + gp) * dx + (1 - 2 * s)


def _chain(args):
    folder, seed, n_draws, n_warmup, cov = args
    _init(folder)
    calls = {"n": 0}

    def counted(u):
        calls["n"] += 1
        return logp_grad(u)

    x0 = np.clip(_sw.mode, _lb + 1e-3, _ub - 1e-3)
    u0 = np.log((x0 - _lb) / (_ub - x0))
    t = time.time()
    r = hmc(
        counted, u0, n_draws, n_chains=1, n_warmup=n_warmup, inv_mass=cov, init_scale=0.02, rng=seed
    )
    draws = np.array([_x(u) for u in r.draws[0]])
    return draws, calls["n"], r.divergences, r.accept_rate, time.time() - t


def main(folder: str, out: str, n_draws: int = 300, n_chains: int = 4) -> None:
    _init(folder)
    d = len(_sw.est)
    x0 = np.clip(_sw.mode, _lb + 1e-3, _ub - 1e-3)
    u0 = np.log((x0 - _lb) / (_ub - x0))
    res = minimize(lambda u: -logp_grad(u)[0], u0, jac=lambda u: -logp_grad(u)[1], method="BFGS")
    umode = res.x
    H = np.empty((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1e-4
        H[i] = (logp_grad(umode + e)[1] - logp_grad(umode - e)[1]) / 2e-4
    w, V = np.linalg.eigh(-0.5 * (H + H.T))
    cov = (V / np.maximum(w, 1e-3)) @ V.T
    print(f"mode found, log posterior {-res.fun:.3f}", flush=True)
    t = time.time()
    with Pool(n_chains) as pool:
        out_ch = pool.map(_chain, [(folder, s, n_draws, 150, cov) for s in range(n_chains)])
    draws = np.stack([c[0] for c in out_ch])
    e = np.array([ess(draws[:, :, i]) for i in range(d)])
    rh = np.array([split_rhat(draws[:, :, i]) for i in range(d)])
    cpu = sum(c[4] for c in out_ch)
    np.savez(out, draws=draws, ess=e, rhat=rh, grads=sum(c[1] for c in out_ch), cpu=cpu)
    print(
        f"done wall {time.time() - t:.0f}s, cpu {cpu:.0f}s, gradients {sum(c[1] for c in out_ch)}, "
        f"divergences {sum(c[2] for c in out_ch)}, accept {np.mean([c[3] for c in out_ch]):.2f}, "
        f"max rhat {rh.max():.3f}, ESS min/median/max {e.min():.0f}/{np.median(e):.0f}/{e.max():.0f}, "
        f"median ESS per CPU-second {np.median(e) / cpu:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 300, int(a[4]) if len(a) > 4 else 4)
