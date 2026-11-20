"""Freeze A17 protocol (PREREGISTRATION_A17). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A17.md", "docs/stage2/OPSIM_SPEC.md", "actionproof/integrity/opsim/__init__.py",
         "actionproof/integrity/opsim/it_ops.py", "actionproof/integrity/opsim/care_admin.py", "scripts/eval_opsim.py",
         "scripts/a17_stats.py", "data/bench/opsim/it_ops.jsonl", "data/bench/opsim/care_admin.jsonl",
         "data/bench/opsim/it_ops_cf.jsonl", "data/bench/opsim/care_admin_cf.jsonl", "tests/integrity/test_opsim_it_ops.py",
         "tests/integrity/test_opsim_care_admin.py", "slurm/tasks/730_a17_main_rv.sh", "slurm/tasks/731_a17_main_ra.sh"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a17_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: A17 protocol matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a17_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
