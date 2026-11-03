"""Compare two benchmark JSON files: ``bench_compare.py old.json new.json --threshold 1.30``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str], float]:
    recs = json.loads(path.read_text(encoding="utf-8"))["records"]
    return {(r["name"], r["backend"]): r["min"] for r in recs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument(
        "--threshold", type=float, default=1.30, help="report slowdowns above this ratio"
    )
    args = ap.parse_args()
    old, new = load(args.old), load(args.new)
    worse = []
    for key in sorted(set(old) & set(new)):
        ratio = new[key] / old[key]
        if ratio > args.threshold:
            worse.append((key, ratio))
            print(
                f"SLOWER {key[0]} [{key[1]}]: {ratio:.2f}x ({old[key] * 1e3:.2f} -> {new[key] * 1e3:.2f} ms)"
            )
    if not worse:
        print("no case slowed down beyond the threshold")
    return 0  # timing is informational, never a gate


if __name__ == "__main__":
    raise SystemExit(main())
