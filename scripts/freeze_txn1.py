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
    ap.add_argument("--addendum1", action="store_true", help="register PREREGISTRATION_TXN1_ADDENDUM1.md (D-TXN1-1)")
    ap.add_argument("--addendum2", action="store_true", help="register PREREGISTRATION_TXN1_ADDENDUM2.md (D-TXN1-2)")
    a = ap.parse_args()
    if a.addendum2:
        dst = ROOT / "results/txn1_addendum2_hashes.json"
        if dst.exists():
            sys.exit(f"{dst} exists; addendum 2 already registered")
        fs = ["PREREGISTRATION_TXN1_ADDENDUM2.md", "scripts/txn_stats.py", "tests/txn/test_scripts.py"]
        dst.write_text(json.dumps({"files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in fs},
                                   "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
        print("registered addendum 2:", len(fs), "files")
        return
    if a.addendum1:
        dst = ROOT / "results/txn1_addendum1_hashes.json"
        if dst.exists():
            sys.exit(f"{dst} exists; addendum 1 already registered")
        fs = ["PREREGISTRATION_TXN1_ADDENDUM1.md", "actionproof/txn/invariants.py", "scripts/txn_invariants_recheck.py",
              "tests/txn/test_policies.py"]
        dst.write_text(json.dumps({"files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in fs},
                                   "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
        print("registered addendum 1:", len(fs), "files")
        return
    ref = ROOT / "results/txn1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        for n in (1, 2):  # files superseded by addendum n (D-TXN1-n), applied in order
            add = ROOT / f"results/txn1_addendum{n}_hashes.json"
            if add.exists():
                old = {**old, **json.loads(add.read_text())["files"]}
        cur.update({f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in old if f not in cur and (ROOT / f).exists()})
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: TXN1 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/txn1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print("registered", len(cur), "files")


if __name__ == "__main__":
    main()
