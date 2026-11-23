"""Freeze algorithm-discovery development round AD-D1, Amendment 1 (results/ad_hashes_a1.json; the original
registration stays in results/ad_hashes.json). --check compares (every Slurm task runs it first)."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md",
         "actionproof/ad/worlds.py", "actionproof/ad/pools.py", "actionproof/ad/rewards.py",
         "actionproof/ad/rollout.py", "actionproof/ad/train.py", "actionproof/ad/evaluate.py",
         "actionproof/ad/scripted.py", "tests/ad/test_ad.py", "scripts/ad/toy_verify_bandit.py",
         "scripts/ad/scripted_frontier.py", "scripts/ad/toy_verify_bandit_v2.py", "scripts/ad/make_tasks.py",
         "slurm/ad_arm.sh",
         "data/bench/proc/cw_train.jsonl", "data/bench/proc/cw_dev.jsonl", "data/bench/proc/cw_conf.jsonl"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/ad_hashes_a1.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: AD-D1 (amendment 1) matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/ad_hashes_a1.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
