"""Freeze RG2 (PREREGISTRATION_RG2.md). No argument: record the hashes of the registered files once (before any
sealed-split output exists). --check compares; every sealed-split run of scripts/rg1_eval.py and
scripts/rg1_policies_eval.py calls it first."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_RG2.md', 'actionproof/recovery_gate/gate.py', 'actionproof/recovery_gate/gate_v2.py',
         'actionproof/recovery_gate/env.py', 'actionproof/recovery_gate/rg2_env.py', 'actionproof/recovery_gate/tasks_rg2.py',
         'actionproof/sel1/env.py', 'actionproof/sel1/policy.py', 'actionproof/sel1/tasks.py', 'scripts/rg2_eval.py',
         'scripts/rg2_policies_eval.py', 'scripts/rg2_stats.py', 'scripts/rg1_eval.py', 'scripts/rg1_stats.py',
         'scripts/txn_eval.py', 'scripts/p1_eval.py', 'actionproof/txn/env.py', 'actionproof/txn/truth.py',
         'actionproof/txn/policies.py', 'actionproof/train/merge.py', 'data/bench/proc/rg2_test.jsonl',
         'data/bench/proc/rg2_manifest.json', 'tests/recovery_gate/test_gate_v2.py']
REF = ROOT / "results/rg2_hashes.json"


def h(f: str) -> str:
    return hashlib.sha256((ROOT / f).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        want = json.loads(REF.read_text())["files"]
        absent = [f for f in want if not (ROOT / f).exists()]
        bad = [f for f in want if f not in absent and h(f) != want[f]]
        print("OK: every RG2 file matches the registration" if not (bad or absent) else
              f"MISMATCH: {bad} ABSENT: {absent}")
        sys.exit(1 if bad or absent else 0)
    if REF.exists():
        sys.exit("results/rg2_hashes.json exists; already registered")
    if any((ROOT / "results/rg2").glob("*__rg2_test.*")):
        sys.exit("a sealed-split RG2 output already exists; registration must precede it")
    REF.write_text(json.dumps({"files": {f: h(f) for f in FILES},
                               "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2) + "\n")
    print("registered", len(FILES), "files")


if __name__ == "__main__":
    main()
