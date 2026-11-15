"""Freeze Amendment 9 (PREREGISTRATION_A9): grounded judge audit and reward.

    python scripts/freeze_a9.py           # register (refuses to overwrite results/a9_hashes.json)
    python scripts/freeze_a9.py --check   # compare
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A9.md", "actionproof/rewards/llm_judge.py", "actionproof/eval/judge_audit.py",
         "actionproof/train/grpo.py", "slurm/grpo_arm.sh", "slurm/tasks/420_a9_grounded_audit.sh",
         "slurm/tasks/421_a9_grounded_judge_s0.sh", "data/curriculum/band_tiny_v2.jsonl"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a9_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Amendment 9 files match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a9_hashes.json exists; Amendment 9 is already registered (deviations go in SPEC §27)")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
