"""Freeze Amendment 8 addendum 2 (PREREGISTRATION_A8_ADDENDUM2): X1 state cell and A7 ack seed 1.

    python scripts/freeze_a8c.py           # register (refuses to overwrite results/a8c_hashes.json)
    python scripts/freeze_a8c.py --check   # compare
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A8_ADDENDUM2.md", "slurm/a8_x1s.sh", "slurm/a7_ack_s1.sh", "slurm/a7_arm.sh",
         "data/bench/a8/x1_task_ids.jsonl", "actionproof/rewards/pcstr.py", "scripts/a8_interaction.py"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a8c_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Amendment 8 addendum 2 files match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a8c_hashes.json exists; Addendum 2 is already registered (deviations go in SPEC §27)")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
