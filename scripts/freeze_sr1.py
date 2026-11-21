"""Freeze SR1. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_SR1.md',
         'actionproof/stage4/surface.py',
         'actionproof/stage4/train_variants.py',
         'scripts/sr1_stats.py',
         'scripts/s4_stats.py',
         'slurm/s3v_arm.sh',
         'slurm/tasks/977_sr1_s0.sh',
         'slurm/tasks/978_sr1_s1.sh',
         'slurm/tasks/979_sr1_s2.sh',
         'slurm/tasks/990_sr1_eval.sh',
         'tests/stage4/test_surface.py']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/sr1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: SR1 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/sr1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
