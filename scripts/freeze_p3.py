"""Freeze P3 (no-std normaliser in arm D). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_P3.md',
         'actionproof/p3/__init__.py',
         'actionproof/p3/train_nostd.py',
         'scripts/p3_stats.py',
         'scripts/p1_eval.py',
         'slurm/p1_evalset.sh',
         'actionproof/stage3/rollout.py',
         'actionproof/stage3/train.py',
         'actionproof/stage3/rewards.py',
         'data/bench/proc/s3_trainb_half.jsonl',
         'data/bench/proc/s3_rec_train_half.jsonl']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/p3_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: P3 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/p3_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
