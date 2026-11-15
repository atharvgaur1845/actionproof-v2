"""Freeze Amendment 10 addendum: environment v2 evaluation.

    python scripts/freeze_a10b.py           # register (refuses to overwrite results/a10b_hashes.json)
    python scripts/freeze_a10b.py --check   # compare
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["PREREGISTRATION_A10_ADDENDUM.md", "actionproof/env/tools_ecom_v2.py", "actionproof/rollout/agenttune_rollout.py",
         "scripts/make_bench_v2.py", "data/bench/v2/test_iid_v2.jsonl", "tests/test_env_v2.py"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a10b_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Amendment 10 addendum files match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a10b_hashes.json exists; Amendment 10 addendum is already registered (deviations go in SPEC §27)")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
