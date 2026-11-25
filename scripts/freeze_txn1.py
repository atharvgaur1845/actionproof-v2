"""Freeze TXN1. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_TXN1.md', 'docs/txn/TXN_SPEC.md', 'actionproof/txn/__init__.py', 'actionproof/txn/env.py',
         'actionproof/txn/truth.py', 'actionproof/txn/invariants.py', 'actionproof/txn/policies.py', 'actionproof/txn/tasks.py',
         'scripts/txn_make_splits.py', 'scripts/txn_eval.py', 'scripts/txn_policies_eval.py', 'scripts/txn_stats.py',
         'scripts/reference_verifier_txn.py', 'scripts/p1_eval.py', 'tests/golden/txn_golden.jsonl',
         'data/bench/proc/txn_dev.jsonl', 'data/bench/proc/txn_val.jsonl', 'data/bench/proc/txn_test.jsonl',
         'data/bench/proc/txn_manifest.json']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/txn1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: TXN1 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/txn1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print("registered", len(cur), "files")


if __name__ == "__main__":
    main()
