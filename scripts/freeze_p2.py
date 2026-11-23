"""Freeze P2 (ledger, guard, full arm set). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_P2.md',
         'actionproof/p2/__init__.py',
         'actionproof/p2/ledger.py',
         'actionproof/p2/guard.py',
         'actionproof/p2/rollout.py',
         'actionproof/p2/train.py',
         'actionproof/p2/build.py',
         'actionproof/reporting/coverage.py',
         'scripts/p2_eval.py',
         'scripts/p2_stats.py',
         'scripts/p2_oracle.py',
         'scripts/p2_equivalence.py',
         'scripts/p2_make_tasks.py',
         'slurm/p2_evalset.sh',
         'tests/p2/test_p2.py',
         'evaluation/llm_judge_committee/committee_p2.py',
         'scripts/p1_eval.py',
         'actionproof/stage3/rollout.py',
         'actionproof/stage3/train.py',
         'slurm/tasks/1180_p2_smoke.sh',
         'slurm/tasks/1181_p2_oracle.sh',
         'slurm/tasks/1182_p2_ewarm.sh',
         'slurm/tasks/1183_p2_e_s0.sh',
         'slurm/tasks/1184_p2_e_s1.sh',
         'slurm/tasks/1185_p2_e_s2.sh',
         'slurm/tasks/1186_p2_dev_b2g.sh',
         'slurm/tasks/1187_p2_test_B2.sh',
         'slurm/tasks/1188_p2_test_Ag.sh',
         'slurm/tasks/1189_p2_test_Cg.sh',
         'slurm/tasks/1190_p2_test_Dg.sh',
         'slurm/tasks/1191_p2_test_E.sh',
         'slurm/tasks/1192_p2_ref.sh',
         'slurm/tasks/1193_p2_shift_A.sh',
         'slurm/tasks/1194_p2_shift_C.sh',
         'slurm/tasks/1195_p2_shift_D.sh',
         'slurm/tasks/1196_p2_shift_E.sh']


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/p2_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: P2 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/p2_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
