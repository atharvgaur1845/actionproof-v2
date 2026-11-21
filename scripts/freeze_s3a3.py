"""Freeze S3 addendum 3. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_S3_ADDENDUM3.md',
         'actionproof/stage4/train_variants.py',
         'scripts/s3_eval_hidden.py',
         'scripts/s3_stats_a3.py',
         'scripts/s4_stats.py',
         'slurm/s3v_arm.sh',
         'slurm/tasks/970_s3a3_stateonly_s0.sh',
         'slurm/tasks/971_s3a3_stateonly_s1.sh',
         'slurm/tasks/972_s3a3_stateonly_s2.sh',
         'slurm/tasks/973_s3a3_stateonly_s3.sh',
         'slurm/tasks/974_s3a3_stateonly_s4.sh',
         'slurm/tasks/975_s3a3_conf.sh',
         'slurm/tasks/976_s3a3_hidden.sh',
         'tests/stage4/test_surface.py']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/s3a3_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: S3 addendum 3 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/s3a3_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
