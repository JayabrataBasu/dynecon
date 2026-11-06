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
from scipy.optimize import minimize
from scipy.special import gammaln

from dynecon.core.errors import DyneconError
from dynecon.solvers.perturbation.dynare import evaluate_assignments, parse_linear_model

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SW:
    """Smets-Wouters (2007): Dynare text -> Dynecon's reader, solver and Kalman likelihood."""

    def __init__(self, folder: str):
        d = Path(folder)
        text = (d / "usmodel.mod").read_text()
        self.model = parse_linear_model(text)
        self.fixed = {"ctou": 0.025, "clandaw": 1.5, "cg": 0.18, "curvp": 10.0, "curvw": 10.0}
        self.stst = (d / "usmodel_stst.m").read_text()
        no_comments = re.sub("//[^" + chr(10) + "]*", "", text)
        est = re.search("estimated_params;(.*?)end;", no_comments, re.S)
        self.est = []
        for ln in est.group(1).strip().splitlines():  # type: ignore[union-attr]
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
        self.init = "dynare2"  # lik_init=2, the replication package's setting
        self.mode = sio.loadmat(d / "usmodel_mode.mat", squeeze_me=True)["xparam1"]

    def params(self, x):
        ns = {**self.fixed, "crhoas": 1.0}
        sd = {}
        for (name, *_), v in zip(self.est, x):
            if name.startswith("stderr_"):
                sd[name[7:]] = v
            else:
                ns[name] = v
        ns.update(evaluate_assignments(self.stst, ns))  # derived steady-state parameters
        return ns, sd

    def loglik(self, x):
        ns, sd = self.params(x)
        return self.model.loglik(self.data, ns, sd, self.obs_names, presample=4, init=self.init)

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
