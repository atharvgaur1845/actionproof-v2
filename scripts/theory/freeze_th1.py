"""Freeze TH1 (PREREGISTRATION_TH1.md). --check compares the current files with the frozen hashes."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["PREREGISTRATION_TH1.md",
         "actionproof/theory/__init__.py",
         "actionproof/theory/exact.py",
         "actionproof/theory/problems.py",
         "actionproof/theory/sim.py",
         "scripts/theory/th1_grid.py",
         "scripts/theory/th1_run.py",
         "scripts/theory/th1_analyze.py",
         "slurm/theory/th1_array.sh",
         "tests/theory/test_exact_theory.py"]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/th1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in FILES if old.get(f) != cur[f]]
        if bad:
            print("MISMATCH:", bad)
            sys.exit(1)
        print("OK: TH1 matches the registration")
        return
    if ref.exists():
        print("refusing to overwrite", ref)
        sys.exit(1)
    ref.write_text(json.dumps({"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                               "files": cur}, indent=1))
    print("wrote", ref)


if __name__ == "__main__":
    main()
