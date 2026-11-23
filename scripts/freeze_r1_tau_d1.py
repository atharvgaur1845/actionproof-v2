"""Freeze R1-T deviation D-R1T-1 (simulated-user server context 32768 -> 8192, Aya Expanse 32B's native limit; the
pilot failed at server start, before any episode). results/r1_tau_d1_hashes.json is kept as registered. --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_R1_TAU.md',
         'actionproof/retention/tau_run.py',
         'actionproof/retention/tau_shim/litellm.py',
         'scripts/r1_tau_stats.py',
         'slurm/r1_tau_set.sh',
         'scripts/freeze_r1_tau.py',
         'slurm/tasks/1104_r1t_pilot.sh',
         'slurm/tasks/1105_r1t_a.sh',
         'slurm/tasks/1106_r1t_b.sh']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/r1_tau_d1_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: R1-T (D-R1T-1) matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/r1_tau_d1_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
