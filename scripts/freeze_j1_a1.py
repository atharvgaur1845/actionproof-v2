"""Freeze J1 addendum 1 (v0 prompt without the warning sentence; exploratory). --check compares.

results/j1_a1_hashes.json is kept as registered."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_J1_ADDENDUM1.md',
         'evaluation/llm_judge_committee/committee.py',
         'evaluation/llm_judge_committee/analyze_a1.py',
         'evaluation/llm_judge_committee/analyze.py',
         'evaluation/llm_judge_committee/data/cases.jsonl',
         'evaluation/llm_judge_committee/data/truth.jsonl']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/j1_a1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: J1 addendum 1 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/j1_a1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
