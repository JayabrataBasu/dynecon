"""Random-walk Metropolis on the Smets-Wouters posterior, for an efficiency comparison with HMC.

Usage: python scripts/real_sw_rw.py DIR OUT.npz [n_draws] [n_chains]
Same unconstrained (logit) scale and log-Jacobian as ``real_sw_sample.py``; one chain per process.
Counts posterior evaluations so effective sample size per evaluation can be compared.
"""

from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from real_sw_estimate import SW

from dynecon.estimation.bayes import ess, metropolis, split_rhat

warnings.filterwarnings("ignore", category=RuntimeWarning)
_sw: SW | None = None
_lb = _ub = None


def _init(folder: str) -> None:
    global _sw, _lb, _ub
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    _sw = SW(folder)
    _lb = np.array([e[2] for e in _sw.est])
    _ub = np.array([e[3] for e in _sw.est])


def _chain(args):
    seed, n_draws, burn_in = args
    calls = {"n": 0}

    def logpost(u):
        calls["n"] += 1
        s = 1 / (1 + np.exp(-u))
        x = _lb + (_ub - _lb) * s
        lj = np.sum(np.log(_ub - _lb) + np.log(s) + np.log1p(-s))
        lp = _sw.logpost(x)
        return float(lp + lj) if lp > -1e9 else float("-inf")

    x0 = np.clip(_sw.mode, _lb + 1e-3, _ub - 1e-3)
    u0 = np.log((x0 - _lb) / (_ub - x0))
    r = metropolis(logpost, u0, n_draws, n_chains=1, burn_in=burn_in, init_scale=0.02, rng=seed)
    return (
        np.array([(_lb + (_ub - _lb) / (1 + np.exp(-u))) for u in r.draws[0]]),
        calls["n"],
        r.accept_rate,
    )


def main(folder: str, out: str, n_draws: int = 40000, n_chains: int = 4) -> None:
    t = time.time()
    with Pool(n_chains, initializer=_init, initargs=(folder,)) as pool:
        res = pool.map(_chain, [(s, n_draws, n_draws // 4) for s in range(n_chains)])
    draws = np.stack([r[0] for r in res])
    evals = sum(r[1] for r in res)
    e = np.array([ess(draws[:, :, i]) for i in range(draws.shape[2])])
    rh = np.array([split_rhat(draws[:, :, i]) for i in range(draws.shape[2])])
    np.savez(out, draws=draws, ess=e, rhat=rh, evals=evals, seconds=time.time() - t)
    print(
        f"done {time.time() - t:.0f}s, evaluations {evals}, accept {np.mean([r[2] for r in res]):.2f}, "
        f"max rhat {rh.max():.3f}, ESS min/median/max {e.min():.0f}/{np.median(e):.0f}/{e.max():.0f}, "
        f"median ESS per 1000 evaluations {np.median(e) / evals * 1000:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 40000, int(a[4]) if len(a) > 4 else 4)
