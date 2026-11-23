"""Freeze AD-C1 (PREREGISTRATION_AD_C1.md) into results/ad_c1_hashes.json. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["PREREGISTRATION_AD_C1.md", "scripts/ad/d1_gates.py", "scripts/ad/d2_gates.py", "scripts/ad/freeze_c1.py",
         "actionproof/ad/evaluate.py", "actionproof/ad/worlds.py", "actionproof/ad/rewards.py",
         "data/bench/proc/cw_conf.jsonl", "scripts/s4_eval.py"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/ad_c1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: AD-C1 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/ad_c1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
