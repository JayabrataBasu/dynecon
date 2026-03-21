"""Fail if deprecated/legacy test folders are used."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    bad_dir = Path("tests/unit")
    if not bad_dir.exists():
        return 0

    bad_files = list(bad_dir.rglob("test_*.py"))
    if bad_files:
        print("Test layout check failed.")
        print("Move tests out of tests/unit into domain folders like:")
        print("- tests/infrastructure/")
        print("- tests/numerics/")
        for path in bad_files:
            print(f"- {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
