"""Freeze P4. --add-data records the built training data's hash once (before training); --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_P4.md', 'actionproof/p4/__init__.py', 'actionproof/p4/build.py', 'scripts/p4_eval.py',
         'scripts/p4_stats.py', 'scripts/p1_eval.py', 'scripts/e1_eval.py', 'actionproof/stage4/surface.py',
         'actionproof/stage4/shift.py', 'actionproof/train/rft.py', 'data/bench/proc/s4_api.jsonl',
         'data/bench/proc/s4_wf4.jsonl', 'data/bench/proc/p1_test_wf4.jsonl']
DATA = ['data/rft/p4_aug_rows.jsonl']


def h(f: str) -> str:
    return hashlib.sha256((ROOT / f).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--add-data", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/p4_hashes.json"
    if a.check:
        reg = json.loads(ref.read_text())
        want = {**reg["files"], **reg.get("data", {})}
        bad = [f for f in want if not (ROOT / f).exists() or h(f) != want[f]]
        print("OK: P4 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if a.add_data:
        reg = json.loads(ref.read_text())
        if reg.get("data"):
            sys.exit("data hash already recorded")
        reg["data"] = {f: h(f) for f in DATA}
        reg["data_recorded"] = datetime.datetime.now().isoformat(timespec="minutes")
        ref.write_text(json.dumps(reg, indent=2))
        print("data recorded", reg["data"])
        return
    if ref.exists():
        sys.exit("results/p4_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": {f: h(f) for f in FILES},
                               "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print("registered", len(FILES), "files")


if __name__ == "__main__":
    main()
