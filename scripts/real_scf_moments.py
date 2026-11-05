"""Weighted wealth moments from an SCF Summary Extract CSV, with a household bootstrap.

Usage: python scripts/real_scf_moments.py PATH_TO_SCFP20XX.csv [out.json]
Moments (all scale-free): net worth / income, Gini of non-negative wealth, bottom-50% share,
top-10% share. ``omega`` is n_households times the bootstrap covariance (sampling error only;
the five SCF implicates are kept together, so imputation variance is ignored).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd


def lorenz_moments(w: np.ndarray, weight: np.ndarray) -> np.ndarray:
    o = np.argsort(w)
    w, wt = w[o], weight[o]
    pop = np.cumsum(wt) / wt.sum()
    cum = np.cumsum(w * wt) / (w * wt).sum()
    pop0, cum0 = np.r_[0.0, pop], np.r_[0.0, cum]
    area = np.sum((cum0[1:] + cum0[:-1]) * np.diff(pop0)) / 2
    return np.array(
        [
            1 - 2 * area,
            np.interp(0.5, pop0, cum0),
            1 - np.interp(0.9, pop0, cum0),
            1 - np.interp(0.99, pop0, cum0),
        ]
    )


def moments(nw: np.ndarray, inc: np.ndarray, wt: np.ndarray) -> np.ndarray:
    ratio = (nw * wt).sum() / (inc * wt).sum()
    return np.r_[ratio, lorenz_moments(np.maximum(nw, 0.0), wt)]


def main(path: str, out: str | None = None, boots: int = 300) -> None:
    d = pd.read_csv(path)
    nw, inc, wt, hh = d.NETWORTH.to_numpy(), d.INCOME.to_numpy(), d.WGT.to_numpy(), d.YY1.to_numpy()
    m = moments(nw, inc, wt)
    ids = np.unique(hh)
    rows = {i: np.flatnonzero(hh == i) for i in ids}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(boots):
        idx = np.concatenate([rows[i] for i in rng.choice(ids, ids.size)])
        draws.append(moments(nw[idx], inc[idx], wt[idx]))
    omega = ids.size * np.cov(np.array(draws).T)
    res = {
        "moments": m.tolist(),
        "omega": omega.tolist(),
        "n_households": int(ids.size),
        "names": ["nw_to_income", "gini", "bottom50_share", "top10_share", "top1_share"],
        "se": np.sqrt(np.diag(omega) / ids.size).tolist(),
        "share_nonpositive_nw": float(wt[nw <= 0].sum() / wt.sum()),
    }
    print(json.dumps(res, indent=1))
    if out:
        with open(out, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
