"""Rule R21b: every name in a public module's __all__ needs a test and a library caller."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = ROOT / "scripts" / "usage_allowlist.txt"


def exported(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return [e.value for e in node.value.elts]  # type: ignore[attr-defined]
    return []


def allowlist() -> set[str]:
    if not ALLOW.exists():
        return set()
    out = set()
    for ln in ALLOW.read_text(encoding="utf-8").splitlines():
        ln = ln.split("#")[0].strip()
        if ln:
            out.add(ln)
    return out


def main() -> int:
    lib = list((ROOT / "econkit").rglob("*.py"))
    tests = list((ROOT / "tests").rglob("*.py"))
    docs = list((ROOT / "docs").rglob("*.md"))
    text = {p: p.read_text(encoding="utf-8") for p in lib + tests + docs}
    allowed = allowlist()
    problems = []
    for path in lib:
        if "backends" in path.parts or path.name == "__init__.py":
            continue
        for name in exported(path):
            if name in allowed:
                continue
            pat = re.compile(rf"\b{re.escape(name)}\b")
            dotted = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            if not any(pat.search(text[t]) or dotted in text[t] for t in tests):
                problems.append(f"{path.relative_to(ROOT)}: {name} has no test reference")
            callers = [
                q for q in lib if q != path and q.name != "__init__.py" and pat.search(text[q])
            ]
            in_docs = any(pat.search(text[d]) for d in docs if "tutorials" in d.parts)
            if not callers and not in_docs:
                problems.append(f"{path.relative_to(ROOT)}: {name} has no library caller")
    if problems:
        print("usage check failed:")
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
