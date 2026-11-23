"""Freeze P1 (practical recovery agent). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_P1.md',
         'actionproof/p1/build.py',
         'scripts/p1_eval.py',
         'scripts/p1_stats.py',
         'slurm/p1_evalset.sh',
         'tests/p1/test_p1.py',
         'actionproof/stage4/progress.py',
         'data/rft/p1_recovery_rows.jsonl',
         'slurm/tasks/1150_p1_c_s0.sh',
         'slurm/tasks/1151_p1_c_s1.sh',
         'slurm/tasks/1152_p1_c_s2.sh',
         'slurm/tasks/1153_p1_dwarm.sh',
         'slurm/tasks/1154_p1_d_s0.sh',
         'slurm/tasks/1155_p1_d_s1.sh',
         'slurm/tasks/1156_p1_d_s2.sh',
         'slurm/tasks/1157_p1_dev_ab.sh',
         'slurm/tasks/1160_p1_test_A_s0.sh',
         'slurm/tasks/1161_p1_test_B_s0.sh',
         'slurm/tasks/1162_p1_test_C_s0.sh',
         'slurm/tasks/1163_p1_test_D_s0.sh',
         'slurm/tasks/1164_p1_test_A_s1.sh',
         'slurm/tasks/1165_p1_test_B_s1.sh',
         'slurm/tasks/1166_p1_test_C_s1.sh',
         'slurm/tasks/1167_p1_test_D_s1.sh',
         'slurm/tasks/1168_p1_test_A_s2.sh',
         'slurm/tasks/1169_p1_test_B_s2.sh',
         'slurm/tasks/1170_p1_test_C_s2.sh',
         'slurm/tasks/1171_p1_test_D_s2.sh',
         'data/bench/proc/p1_dev_full.jsonl',
         'data/bench/proc/p1_dev_rec.jsonl',
         'data/bench/proc/p1_test_combo.jsonl',
         'data/bench/proc/p1_test_full.jsonl',
         'data/bench/proc/p1_test_pp.jsonl',
         'data/bench/proc/p1_test_rec.jsonl',
         'data/bench/proc/p1_test_wf4.jsonl',
         'data/bench/proc/p1_train_full.jsonl',
         'data/bench/proc/p1_train_pp.jsonl',
         'data/bench/proc/p1_train_rec.jsonl']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/p1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: P1 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/p1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
