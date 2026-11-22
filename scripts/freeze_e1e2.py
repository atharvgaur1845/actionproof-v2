"""Freeze E1/E2. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_E1E2.md',
         'scripts/e1e2_make_splits.py',
         'scripts/e1_eval.py',
         'scripts/e2_probe.py',
         'scripts/e1e2_stats.py',
         'scripts/s3_eval.py',
         'scripts/s4_stats.py',
         'data/bench/proc/e1_pp_full.jsonl',
         'data/bench/proc/e1_pp_rec.jsonl',
         'data/bench/proc/e2_pp.jsonl',
         'slurm/e1e2_set.sh',
         'slurm/tasks/1001_e1e2_ref.sh',
         'slurm/tasks/1002_e1e2_rsc.sh',
         'slurm/tasks/1003_e1e2_ctl.sh',
         'slurm/tasks/1004_e1e2_ack.sh']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/e1e2_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: E1/E2 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/e1e2_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
