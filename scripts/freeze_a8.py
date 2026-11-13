"""Freeze Amendment 8 (PREREGISTRATION_A8): X1 task list, audit script and output, X1 run script, rollout logger.

    python scripts/freeze_a8.py           # register (refuses to overwrite results/a8_hashes.json)
    python scripts/freeze_a8.py --check   # compare
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A8.md", "data/bench/a8/x1_task_ids.jsonl", "scripts/a8_ack_class_audit.py",
         "results/a8/ack_class_audit.md", "slurm/a8_x1.sh", "actionproof/rollout/batched_vllm.py",
         "actionproof/rewards/ack_outcome.py", "data/bench/a4/a4_train_B.jsonl"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a8_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Amendment 8 files match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a8_hashes.json exists; Amendment 8 is already registered (deviations go in SPEC §27)")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
