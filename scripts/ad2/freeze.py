"""Hash-freeze a v2 registration: python scripts/ad2/freeze.py <name> [--check].

The file list is read from the registration's own `FILES:` block in docs/research/algorithm_discovery_v2/<name>.md
(lines "- path" until a blank line), so the registration document names exactly what it freezes.
Hashes go to results/ad2/hashes_<name>.json and are never overwritten.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def files_of(name: str) -> list[str]:
    doc = ROOT / "docs/research/algorithm_discovery_v2" / f"{name}.md"
    out, on = [doc.relative_to(ROOT).as_posix()], False
    for line in doc.read_text().splitlines():
        if line.strip() == "FILES:":
            on = True
            continue
        if on:
            if not line.strip():
                break
            out.append(line.strip().lstrip("- ").strip("`"))
    return out


def main() -> None:
    name = sys.argv[1]
    ref = ROOT / "results/ad2" / f"hashes_{name}.json"
    cur = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files_of(name)}
    if "--check" in sys.argv:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print(f"OK: {name} matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit(f"{ref} exists; already registered")
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
