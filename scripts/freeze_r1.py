"""Freeze R1 (capability retention). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_R1.md',
         'actionproof/retention/build_prompts.py',
         'actionproof/retention/gen.py',
         'scripts/r1_score.py',
         'scripts/r1_stats.py',
         'slurm/r1_evalset.sh',
         'slurm/tasks/1099_r1_smoke.sh',
         'slurm/tasks/1100_r1_a.sh',
         'slurm/tasks/1101_r1_b.sh',
         'slurm/tasks/1102_r1_c.sh',
         'slurm/tasks/1103_r1_d.sh',
         'data/retention/r1_inputs.jsonl',
         'data/retention/ifeval_input_data.jsonl',
         'configs/r1_scoring_requirements.txt',
         'tests/retention/test_r1.py']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/r1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: R1 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/r1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
