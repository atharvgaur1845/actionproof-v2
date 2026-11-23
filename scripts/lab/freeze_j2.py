"""Freeze J2 (PREREGISTRATION_J2.md). --check compares the current files with the frozen hashes."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "PREREGISTRATION_J2.md"
REF = ROOT / "results/j2_hashes.json"


def files() -> list[str]:
    txt = REG.read_text().split("FILES:", 1)[1]
    return [l.strip()[2:] for l in txt.splitlines() if l.strip().startswith("- ")]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    cur = derive()
    if a.check:
        ref = json.loads(REF.read_text())["files"]
        bad = [f for f in ref if ref[f] != cur.get(f)]
        if bad:
            print("MISMATCH:", bad)
            sys.exit(1)
        print("OK: J2 matches the registration")
        return
    if REF.exists():
        sys.exit(f"{REF} exists: J2 is already frozen (amend with a dated addendum)")
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    REF.write_text(json.dumps({"registered_at": now.isoformat(timespec="seconds"), "files": cur}, indent=1))
    print("frozen", now.isoformat(timespec="seconds"), len(cur), "files")


if __name__ == "__main__":
    main()
