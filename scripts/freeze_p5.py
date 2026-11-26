"""Freeze P5. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_P5.md', 'scripts/p5_eval.py', 'scripts/p5_stats.py', 'scripts/p1_eval.py', 'actionproof/train/rft.py',
         'actionproof/train/merge.py', 'data/rft/p1_dwarm_rows.jsonl.sha256']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--addendum1", action="store_true", help="register PREREGISTRATION_P5_ADDENDUM1.md (D-P5-1)")
    a = ap.parse_args()
    if a.addendum1:
        dst = ROOT / "results/p5_addendum1_hashes.json"
        if dst.exists():
            sys.exit(f"{dst} exists; addendum 1 already registered")
        fs = ["PREREGISTRATION_P5_ADDENDUM1.md", "scripts/p5_eval.py", "tests/p5/test_p5_eval.py"]
        dst.write_text(json.dumps({"files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in fs},
                                   "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
        print("registered addendum 1:", len(fs), "files")
        return
    ref = ROOT / "results/p5_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        add = ROOT / "results/p5_addendum1_hashes.json"  # files superseded by addendum 1 (D-P5-1)
        if add.exists():
            old = {**old, **json.loads(add.read_text())["files"]}
        cur.update({f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in old if f not in cur and (ROOT / f).exists()})
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: P5 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/p5_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print("registered", len(cur), "files")


if __name__ == "__main__":
    main()
