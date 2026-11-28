"""Freeze AD-C2 (PREREGISTRATION_AD_C2.md) into results/ad_c2_hashes.json. --check compares.
Refuses to register if any cw_c2 model output already exists."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["PREREGISTRATION_AD_C2.md", "scripts/ad/freeze_c2.py", "scripts/ad/c2_split.py", "scripts/ad/c2_stats.py",
         "actionproof/ad/train.py", "actionproof/ad/rewards.py", "actionproof/ad/evaluate.py", "actionproof/ad/worlds.py",
         "actionproof/ad/pools.py", "actionproof/ad/rollout.py", "slurm/ad_arm_c2.sh",
         "slurm/tasks/2000_c2_grpo_c001.sh", "slurm/tasks/2001_c2_nostd_c001.sh", "slurm/tasks/2002_c2_grpo_c000.sh",
         "slurm/tasks/2003_c2_eval_stats.sh", "data/bench/proc/cw_c2.jsonl", "data/bench/proc/cw_train.jsonl"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/ad_c2_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: AD-C2 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/ad_c2_hashes.json exists; already registered")
    if list((ROOT / "results/ad/c2").glob("*cw_c2*")):
        sys.exit("cw_c2 outputs exist; cannot register")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
