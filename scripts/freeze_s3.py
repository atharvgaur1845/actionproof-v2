"""Freeze S3 protocol (PREREGISTRATION_S3). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_S3.md',
         'actionproof/stage3/__init__.py',
         'actionproof/stage3/env.py',
         'actionproof/stage3/rewards.py',
         'actionproof/stage3/tasks.py',
         'actionproof/stage3/rollout.py',
         'actionproof/stage3/train.py',
         'actionproof/stage3/scripted.py',
         'scripts/s3_eval.py',
         'scripts/s3_stats.py',
         'scripts/eval_opsim.py',
         'slurm/s3_arm.sh',
         'slurm/s3_conf.sh',
         'data/bench/proc/s3_trainb_half.jsonl',
         'data/bench/proc/s3_rec_train_half.jsonl',
         'data/bench/proc/s3_conf_rec.jsonl',
         'data/bench/proc/s3_conf_full.jsonl',
         'data/bench/proc/s3_conf_stress.jsonl',
         'data/bench/opsim/s3_it_ops.jsonl',
         'data/bench/opsim/s3_care_admin.jsonl',
         'results/s3/scripted_conf_calibration.json']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/s3_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: S3 protocol matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/s3_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
