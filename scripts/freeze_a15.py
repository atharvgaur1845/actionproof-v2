"""Freeze A15/A16 protocol (PREREGISTRATION_A15). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A15.md", "scripts/a15_eval.py", "scripts/a16_mech.py", "scripts/cef_eval.py",
         "data/bench/proc/proc_mech.jsonl", "data/bench/proc/proc_mech_dev.jsonl", "tests/integrity/test_a15.py",
         "slurm/tasks/710_a15_main_rv.sh", "slurm/tasks/711_a15_main_ra.sh", "slurm/tasks/712_a16_traj_rv.sh",
         "slurm/tasks/713_a16_traj_ra.sh", "slurm/tasks/714_a16_causal_a.sh", "slurm/tasks/715_a16_causal_b.sh"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a15_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: A15/A16 protocol matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a15_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
