"""Freeze Stage 5 development (C5, G). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['docs/research/algorithm/C5G_DEV.md',
         'actionproof/stage4/progress.py',
         'actionproof/stage4/rollout_c5g.py',
         'actionproof/stage4/train_c5g.py',
         'tests/stage4/test_c5g.py',
         'scripts/s5_dev.py',
         'scripts/s3_eval.py',
         'slurm/s5_arm.sh',
         'slurm/tasks/1009_c5g_smoke.sh',
         'slurm/tasks/1010_c5g_c5_s0.sh',
         'slurm/tasks/1011_c5g_guide_s0.sh',
         'slurm/tasks/1012_c5g_c5_s1.sh',
         'slurm/tasks/1013_c5g_guide_s1.sh',
         'data/bench/proc/s3_trainb_half.jsonl',
         'data/bench/proc/s3_rec_train_half.jsonl',
         'data/bench/proc/s3_rec_dev.jsonl',
         'data/bench/proc/proc_dev.jsonl']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/c5g_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Stage 5 development (C5, G) matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/c5g_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
