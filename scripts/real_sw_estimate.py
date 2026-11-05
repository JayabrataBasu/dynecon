"""Smets-Wouters (2007) US model: read the Dynare ``usmodel.mod``, solve, filter, find the mode.

Usage: python scripts/real_sw_estimate.py DIR   (DIR = the openICPSR 116269-V1 folder)
Reads the linear model equations from the .mod text, solves them with Dynecon's Klein solver,
evaluates the Gaussian likelihood with Dynecon's Kalman filter (stationary initial state, first
four periods as presample, as in the replication package), adds the priors from the .mod, and
compares the posterior mode with the mode Dynare found (usmodel_mode.mat).
"""

from __future__ import annotations

import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.special import gammaln

from dynecon.core.errors import DyneconError
from dynecon.estimation.kalman import kalman_filter
from dynecon.solvers.perturbation.klein import klein_solve

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Lin:
    """Linear expression in (variable, lag) pairs plus a constant; ``x(-1)`` shifts the lag."""

    def __init__(self, c=None, k=0.0):
        self.c, self.k = dict(c or {}), k

    def __call__(self, s):
        return Lin({(n, lag + s): v for (n, lag), v in self.c.items()}, self.k)

    def _o(self, o):
        return o if isinstance(o, Lin) else Lin({}, float(o))

    def __add__(self, o):
        o = self._o(o)
        c = dict(self.c)
        for key, v in o.c.items():
            c[key] = c.get(key, 0.0) + v
        return Lin(c, self.k + o.k)

    __radd__ = __add__

    def __neg__(self):
        return Lin({key: -v for key, v in self.c.items()}, -self.k)

    def __sub__(self, o):
        return self + (-self._o(o))

    def __rsub__(self, o):
        return self._o(o) + (-self)

    def __mul__(self, a):
        assert not isinstance(a, Lin)
        return Lin({key: v * a for key, v in self.c.items()}, self.k * a)

    __rmul__ = __mul__

    def __truediv__(self, a):
        return self * (1.0 / a)


class SW:
    def __init__(self, folder: str):
        d = Path(folder)
        text = re.sub(r"//[^\n]*", "", (d / "usmodel.mod").read_text())
        self.vars = re.search(r"\bvar\s+(.*?);", text, re.S).group(1).split()
        self.exo = re.search(r"\bvarexo\s+(.*?);", text, re.S).group(1).split()
        body = re.search(r"model\(linear\);(.*?)\bend;", text, re.S).group(1)
        body = body.replace("#usmodel_stst;", "")
        self.eqs = [" ".join(e.split()).replace("^", "**") for e in body.split(";") if "=" in e]
        self.fixed = {"ctou": 0.025, "clandaw": 1.5, "cg": 0.18, "curvp": 10.0, "curvw": 10.0}
        stst = re.sub(r"%[^\n]*", "", (d / "usmodel_stst.m").read_text())
        self.stst = stst.replace("^", "**")
        est = re.search(r"estimated_params;(.*?)\bend;", text, re.S).group(1)
        self.est = []
        for ln in est.strip().splitlines():
            ln = ln.strip().rstrip(";")
            if not ln:
                continue
            f = [x.strip() for x in ln.replace("stderr ", "stderr_").split(",")]
            self.est.append(
                (f[0], float(f[1]), float(f[2]), float(f[3]), f[4], float(f[5]), float(f[6]))
            )
        mat = sio.loadmat(d / "usmodel_data.mat", squeeze_me=True)
        self.obs_names = ["dy", "dc", "dinve", "labobs", "pinfobs", "dw", "robs"]
        self.data = np.column_stack([mat[k] for k in self.obs_names])[70:]  # first_obs = 71
        self.init = 2  # Dynare lik_init=2 (the replication package's setting)
        self.mode = sio.loadmat(d / "usmodel_mode.mat", squeeze_me=True)["xparam1"]

    def params(self, x):
        ns = dict(self.fixed)
        ns.update(crhoas=1.0)
        shock_sd = {}
        for (name, *_), v in zip(self.est, x):
            if name.startswith("stderr_"):
                shock_sd[name[7:]] = v
            else:
                ns[name] = v
        for line in self.stst.split(";"):
            line = line.strip()
            if "=" in line:
                exec(line, {}, ns)  # noqa: S102 - the .mod is the user's own file
        return ns, shock_sd

    def matrices(self, x):
        ns, sd = self.params(x)
        n, m = len(self.vars), len(self.exo)
        env = {v: Lin({(v, 0): 1.0}) for v in self.vars}
        env.update({e: Lin({(e, 0): 1.0}) for e in self.exo})
        env.update(ns)
        C0, C1, C2, D = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, m))
        const = np.zeros(n)
        vi, ei = {v: i for i, v in enumerate(self.vars)}, {e: i for i, e in enumerate(self.exo)}
        assert len(self.eqs) == n, (len(self.eqs), n)
        for r, eq in enumerate(self.eqs):
            lhs, rhs = eq.split("=")
            e = eval(lhs, {"__builtins__": {}}, env) - eval(rhs, {"__builtins__": {}}, env)  # noqa: S307
            e = e if isinstance(e, Lin) else Lin({}, float(e))
            for (name, lag), v in e.c.items():
                if name in ei:
                    D[r, ei[name]] += v
                else:
                    {1: C0, 0: C1, -1: C2}[lag][r, vi[name]] += v
            const[r] = e.k
        return C0, C1, C2, D, const, np.array([sd[e] for e in self.exo])

    def state_space(self, x):
        C0, C1, C2, D, const, sd = self.matrices(x)
        n, m = len(self.vars), len(self.exo)
        N = 2 * n + m
        A, B = np.zeros((N, N)), np.zeros((N, N))
        # w = [y_{t-1} (n); eps_t (m); y_t (n)], A E[w_{t+1}] = B w_t, first n+m predetermined
        A[:n, :n] = np.eye(n)
        B[:n, n + m :] = np.eye(n)  # y_{t} (next period's lag) = jump variable y_t
        A[n : n + m, n : n + m] = np.eye(m)  # E[eps_{t+1}] = 0
        A[n + m :, n + m :] = C0
        B[n + m :, n + m :] = -C1
        B[n + m :, :n] = -C2
        B[n + m :, n : n + m] = -D
        F, P, _ = klein_solve(A, B, n + m)
        R = np.zeros((n + m, m))
        R[n:, :] = np.diag(sd)
        rows = [self.vars.index(o) for o in self.obs_names]
        lhs = [e.split("=")[0].strip() for e in self.eqs]  # equation order differs from variables
        d = np.array([-const[lhs.index(o)] for o in self.obs_names])  # steady state = constant
        lagged = np.abs(C2).sum(axis=0) > 0  # variables that appear with a lag: Dynare's states
        P0 = np.zeros((n + m, n + m))  # lik_init=2: variance 10 on those states
        P0[:n, :n] = np.diag(10.0 * lagged)
        P0[n:, n:] = np.diag(sd**2)  # the first period's innovations
        self.P0 = P0
        Zf = F[rows]
        keep = (
            np.abs(Zf).sum(axis=0) > 0
        )  # observability closure: states that can reach an observable
        while True:
            more = keep | (np.abs(P[keep]).sum(axis=0) > 0)
            if (more == keep).all():
                break
            keep = more
        ix = np.flatnonzero(keep)
        self.P0 = self.P0[np.ix_(ix, ix)]
        return Zf[:, ix], P[np.ix_(ix, ix)], (R @ R.T)[np.ix_(ix, ix)], d

    def loglik(self, x):
        Z, P, Q, d = self.state_space(x)
        y = self.data - d
        P0 = self.P0 if self.init == 2 else solve_discrete_lyapunov(P, Q)
        H, a0 = np.zeros((7, 7)), np.zeros(P.shape[0])
        full = kalman_filter(y, Z, H, P, Q, a0, P0).loglik
        return full - kalman_filter(y[:4], Z, H, P, Q, a0, P0).loglik  # presample = 4

    def logprior(self, x):
        lp = 0.0
        for v, (_, _, lb, ub, shape, p1, p2) in zip(x, self.est):
            if not lb < v < ub:
                return -math.inf
            if shape == "NORMAL_PDF":
                lp += -0.5 * ((v - p1) / p2) ** 2 - math.log(p2 * math.sqrt(2 * math.pi))
            elif shape == "BETA_PDF":
                a = p1 * ((p1 * (1 - p1)) / p2**2 - 1)
                b = a * (1 - p1) / p1
                lp += (a - 1) * math.log(v) + (b - 1) * math.log(1 - v)
                lp -= gammaln(a) + gammaln(b) - gammaln(a + b)
            elif shape == "GAMMA_PDF":
                k, th = (p1 / p2) ** 2, p2**2 / p1
                lp += (k - 1) * math.log(v) - v / th - gammaln(k) - k * math.log(th)
            elif (
                shape == "INV_GAMMA_PDF"
            ):  # Smets-Wouters: scale s = p1, degrees of freedom nu = p2
                s, nu = p1, p2
                lp += math.log(2) - gammaln(nu / 2) + (nu / 2) * math.log(nu * s * s / 2)
                lp += -(nu + 1) * math.log(v) - nu * s * s / (2 * v * v)
            else:
                raise ValueError(shape)
        return lp

    def logpost(self, x):
        lp = self.logprior(x)
        if not np.isfinite(lp):
            return -1e10
        try:
            ll = self.loglik(x)
        except (DyneconError, np.linalg.LinAlgError, ValueError):
            return -1e10
        return lp + ll if np.isfinite(ll) else -1e10


def main(folder: str) -> None:
    sw = SW(folder)
    x0 = np.array([e[1] for e in sw.est])
    print("observations:", sw.data.shape)
    print(
        f"loglik at Dynare mode {sw.loglik(sw.mode):.3f}   log posterior {sw.logpost(sw.mode):.3f}"
    )
    print(f"log posterior at .mod initial values {sw.logpost(x0):.3f}")
    bounds = [(e[2] + 1e-6, e[3] - 1e-6) for e in sw.est]
    opts = {"maxfun": 60000, "maxiter": 2000}
    res = minimize(lambda x: -sw.logpost(x), x0, method="L-BFGS-B", bounds=bounds, options=opts)
    nm = {"maxfev": 20000, "xatol": 1e-6, "fatol": 1e-8}
    res = minimize(lambda x: -sw.logpost(x), res.x, method="Nelder-Mead", options=nm)
    print(f"our maximum log posterior {-res.fun:.3f}")
    print(f"{'param':14s}{'ours':>10s}{'Dynare':>10s}")
    for e, a, b in zip(sw.est, res.x, sw.mode):
        print(f"{e[0]:14s}{a:10.4f}{b:10.4f}")
    print("max |diff|:", np.abs(res.x - sw.mode).max())


if __name__ == "__main__":
    main(sys.argv[1])
