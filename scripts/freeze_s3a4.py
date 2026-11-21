"""Freeze S3 addendum 4. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_S3_ADDENDUM4.md',
         'scripts/s3_make_ackonly.py',
         'scripts/s3_stats_a4.py',
         'scripts/s4_stats.py',
         'data/bench/proc/s3_rec_train_half_ackonly.jsonl',
         'slurm/s3_arm.sh',
         'slurm/s3_ppset.sh',
         'slurm/tasks/991_s3a4_ackonly_s0.sh',
         'slurm/tasks/992_s3a4_ackonly_s1.sh',
         'slurm/tasks/993_s3a4_ackonly_s2.sh',
         'slurm/tasks/994_s3a4_ackonly_s3.sh',
         'slurm/tasks/995_s3a4_ackonly_s4.sh',
         'slurm/tasks/996_s3a4_conf.sh',
         'slurm/tasks/997_s3a4_pp.sh']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/s3a4_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: S3 addendum 4 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/s3a4_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
