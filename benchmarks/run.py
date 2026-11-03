"""Run the benchmark suite: ``python -m benchmarks.run --all --out benchmarks/results/<file>.json``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import cases as case_module
from benchmarks.harness import machine_info, run_case
from dynecon import backends


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run every case")
    ap.add_argument("--case", action="append", default=[], help="case name (repeatable)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 repeats (for the nightly job)")
    args = ap.parse_args(argv)
    factories = {f.__name__: f for f in case_module.ALL}
    names = list(factories) if args.all else args.case
    if not names:
        ap.error("give --all or --case NAME")
    repeat = 2 if args.quick else args.repeat
    records: list[dict[str, Any]] = []
    for name in names:
        case = factories[name]()
        state = case.setup()
        reference: np.ndarray | None = None
        for backend in case.backends:
            if not backends.is_available(backend):
                continue
            rec, result = run_case(case, backend, reference, n_repeat=repeat, state=state)
            if backend == "numpy":
                reference = result
            records.append(rec)
            print(
                f"{name:18s} {backend:6s} min {rec['min'] * 1e3:10.3f} ms   compile {rec['compile_seconds']:.2f} s"
            )
    out = (
        args.out or Path("benchmarks/results") / f"{dt.date.today():%Y%m%d}_{platform.node()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"machine": machine_info(), "date": str(dt.date.today()), "records": records}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
