"""Freeze SEL1 (PREREGISTRATION_SEL1.md). No argument: record the hashes of the registered files once.
--add-data records the built training data's hashes once (built on the cluster, before training). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_SEL1.md', 'actionproof/sel1/__init__.py', 'actionproof/sel1/env.py',
         'actionproof/sel1/tasks.py', 'actionproof/sel1/policy.py', 'actionproof/sel1/build.py',
         'scripts/sel1_eval.py', 'scripts/sel1_stats.py', 'scripts/sel1_policies_eval.py', 'scripts/txn_eval.py',
         'scripts/p1_eval.py', 'actionproof/txn/env.py', 'actionproof/txn/truth.py', 'actionproof/txn/tasks.py',
         'actionproof/txn/policies.py', 'actionproof/txn/invariants.py', 'actionproof/train/rft.py',
         'actionproof/train/merge.py', 'data/bench/proc/sel1_train.jsonl', 'data/bench/proc/sel1_ctl_pool.jsonl',
         'data/bench/proc/sel1_test.jsonl', 'data/bench/proc/sel1_manifest.json', 'data/bench/proc/txn_test.jsonl',
         'data/rft/p4_aug_rows.jsonl', 'tests/sel1/test_sel1.py']
DATA = ['data/rft/sel1_sel_rows.jsonl', 'data/rft/sel1_ctl_rows.jsonl', 'data/rft/sel1_rows.stats.json']


def h(f: str) -> str:
    return hashlib.sha256((ROOT / f).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--add-data", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/sel1_hashes.json"
    if a.check:
        reg = json.loads(ref.read_text())
        want = {**reg["files"], **reg.get("data", {})}
        absent = [f for f in want if not (ROOT / f).exists()]
        bad = [f for f in want if f not in absent and h(f) != want[f]]
        if absent:  # the training rows (about 61 MB each) are in the traces dataset, not in git
            print("ABSENT locally (download from atharvgaur18/actionproof-traces, then re-check): " + ", ".join(absent))
        print("OK: every present SEL1 file matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad or absent else 0)
    if a.add_data:
        reg = json.loads(ref.read_text())
        if reg.get("data"):
            sys.exit("data hashes already recorded")
        reg["data"] = {f: h(f) for f in DATA}
        reg["data_recorded"] = datetime.datetime.now().isoformat(timespec="minutes")
        ref.write_text(json.dumps(reg, indent=2))
        print("data recorded", reg["data"])
        return
    if ref.exists():
        sys.exit("results/sel1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": {f: h(f) for f in FILES},
                               "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print("registered", len(FILES), "files")


if __name__ == "__main__":
    main()
