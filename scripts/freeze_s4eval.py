"""Freeze S4-EVAL shift evaluation (PREREGISTRATION_S4_EVAL). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_S4_EVAL.md',
         'actionproof/stage4/shift.py',
         'scripts/s4_make_splits.py',
         'scripts/s4_eval.py',
         'scripts/s3_eval.py',
         'scripts/s3_stats.py',
         'tests/stage4/test_shift.py',
         'data/bench/proc/s4_lang_en.jsonl',
         'data/bench/proc/s4_lang_hi.jsonl',
         'data/bench/proc/s4_lang_hinglish.jsonl',
         'data/bench/proc/s4_lang_para.jsonl',
         'data/bench/proc/s4_api.jsonl',
         'data/bench/proc/s4_wf4.jsonl',
         'data/bench/proc/s4_combo.jsonl']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/s4eval_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: S4-EVAL protocol matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/s4eval_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
