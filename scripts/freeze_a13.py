"""Freeze Amendment 13 (PREREGISTRATION_A13). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A13.md", "actionproof/integrity/clauses.py", "scripts/h5_coverage.py", "scripts/h7_matrix.py",
         "tests/adversarial/proc_adversarial.jsonl", "tests/adversarial/README.md", "data/bench/proc/proc_dev.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a13_hashes.json"
    cur = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Amendment 13 files match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a13_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(len(cur), "files hashed")


if __name__ == "__main__":
    main()
