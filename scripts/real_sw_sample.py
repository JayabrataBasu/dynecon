"""Sample the Smets-Wouters posterior with Dynecon's HMC (finite-difference gradient, parallel).

Usage: python scripts/real_sw_sample.py DIR OUT.npz [n_draws] [n_chains]
Needs the posterior mode first (re-computed here from the Dynare mode as a start). Sampling is
on an unconstrained logit scale; the mass matrix is the inverse Hessian at the mode.
"""

from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from real_sw_estimate import SW
from scipy.optimize import minimize

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


def _f(u):
    """Log posterior plus log-Jacobian of the logit map, at unconstrained ``u``."""
    s = 1 / (1 + np.exp(-u))
    lj = np.sum(np.log(_ub - _lb) + np.log(s) + np.log1p(-s))
    return _sw.logpost(_x(u)) + lj


def _grad_chunk(args):
    u, idx, h = args
    out = []
    for i in idx:
        e = np.zeros_like(u)
        e[i] = h
        out.append((_f(u + e) - _f(u - e)) / (2 * h))
    return out


def main(folder: str, out: str, n_draws: int = 400, n_chains: int = 4) -> None:
    _init(folder)
    d = len(_sw.est)
    with Pool(12, initializer=_init, initargs=(folder,)) as pool:
        chunks = np.array_split(np.arange(d), 12)

        def logp_grad(u):
            lp = _f(u)
            if not np.isfinite(lp) or lp < -1e9:
                return float("-inf"), np.zeros(d)
            parts = pool.map(_grad_chunk, [(u, c, 1e-5) for c in chunks])
            return lp, np.array([g for p in parts for g in p])

        x0 = np.clip(_sw.mode, _lb + 1e-3, _ub - 1e-3)
        u0 = np.log((x0 - _lb) / (_ub - x0))
        res = minimize(
            lambda u: -logp_grad(u)[0], u0, jac=lambda u: -logp_grad(u)[1], method="BFGS"
        )
        umode = res.x
        print(f"mode log posterior (unconstrained, incl. Jacobian) {-res.fun:.3f}", flush=True)
        # Hessian by differencing the analytic-quality FD gradient
        H = np.empty((d, d))
        for i in range(d):
            e = np.zeros(d)
            e[i] = 1e-4
            H[i] = (logp_grad(umode + e)[1] - logp_grad(umode - e)[1]) / 2e-4
        H = -0.5 * (H + H.T)
        w, V = np.linalg.eigh(H)
        print("Hessian eigenvalues min/max", w.min(), w.max(), flush=True)
        cov = (V / np.maximum(w, 1e-3)) @ V.T
        t = time.time()
        calls = {"n": 0}

        def counted(u):
            calls["n"] += 1
            if calls["n"] % 200 == 0:  # progress: gradients so far and elapsed time
                print(f"  {calls['n']} gradients, {time.time() - t:.0f}s", flush=True)
            return logp_grad(u)

        r = hmc(
            counted,
            umode,
            n_draws,
            n_chains=n_chains,
            n_warmup=150,
            inv_mass=cov,
            init_scale=0.02,
            rng=1,
            names=[e[0] for e in _sw.est],
        )
        draws = np.array([[_x(u) for u in ch] for ch in r.draws])
        np.savez(
            out,
            draws=draws,
            mode=_x(umode),
            rhat=r.rhat,
            ess=r.ess,
            divergences=r.divergences,
            accept=r.accept_rate,
            grads=calls["n"],
            seconds=time.time() - t,
        )
        print(
            f"done {time.time() - t:.0f}s, gradients {calls['n']}, accept {r.accept_rate:.2f}, "
            f"divergences {r.divergences}, max rhat {r.rhat.max():.3f}, min ess {r.ess.min():.0f}",
            flush=True,
        )


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 400, int(a[4]) if len(a) > 4 else 4)
