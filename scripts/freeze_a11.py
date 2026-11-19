"""Freeze Amendment 11 (PREREGISTRATION_A11): c2 splits, taxonomy labels and script, e-commerce gateway, eval runner.

    python scripts/freeze_a11.py           # register (refuses to overwrite results/a11_hashes.json)
    python scripts/freeze_a11.py --check   # compare; exits 1 on any mismatch or missing registration
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "PREREGISTRATION_A11.md",
    "data/bench/c2/test_iid_c2.jsonl",
    "data/bench/c2/v2_stress_c2.jsonl",
    "data/bench/c2/fault_labels.jsonl",
    "scripts/classify_faults.py",
    "scripts/make_splits_c2.py",
    "actionproof/integrity/adapters/ecom.py",
    "scripts/eval_gateway.py",
]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/a11_hashes.json"
    if a.check:
        if not ref.exists():
            print("MISMATCH: results/a11_hashes.json does not exist (Amendment 11 is not registered)")
            sys.exit(1)
        old = json.loads(ref.read_text())["files"]
        cur = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() if (ROOT / f).exists() else None for f in old}
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: Amendment 11 files match the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/a11_hashes.json exists; Amendment 11 is already registered (deviations go in SPEC §27)")
    if "DRAFT" in (ROOT / "PREREGISTRATION_A11.md").read_text().splitlines()[0]:
        sys.exit("PREREGISTRATION_A11.md is still marked DRAFT on its first line; finalize it before registering")
    cur = derive()
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")},
                              indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
