"""Freeze S3 addendum 2 (PREREGISTRATION_S3_ADDENDUM2). --check compares."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['PREREGISTRATION_S3_ADDENDUM2.md',
         'scripts/s3_make_rebal.py',
         'scripts/s3_stats_a2.py',
         'scripts/s3_zero_var.py',
         'tests/stage3/test_rebal.py',
         'scripts/s4_stats.py',
         'slurm/s3_ppset.sh',
         'data/bench/proc/s3_trainb_half_rb.jsonl',
         'data/bench/proc/s3_rec_train_half_rb.jsonl',
         'data/bench/proc/s3_matched_full_half_rb.jsonl',
         'data/bench/proc/s3_pp_full.jsonl',
         'data/bench/proc/s3_pp_rec.jsonl',
         'results/s3/scripted_pp_calibration.json'] + [
    f'slurm/tasks/{t}.sh' for t in ('960_s3a2_matched_s3', '961_s3a2_matched_s4', '962_s3a2_rb_mix_s0',
                                    '963_s3a2_rb_mix_s1', '964_s3a2_rb_matched_s0', '965_s3a2_rb_matched_s1',
                                    '966_s3a2_conf_matched', '967_s3a2_pp_base', '968_s3a2_pp_new', '969_s3a2_conf_rb')]


def derive() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ref = ROOT / "results/s3a2_hashes.json"
    cur = derive()
    if a.check:
        old = json.loads(ref.read_text())["files"]
        bad = [f for f in old if cur.get(f) != old[f]]
        print("OK: S3 addendum 2 matches the registration" if not bad else "MISMATCH: " + ", ".join(bad))
        sys.exit(1 if bad else 0)
    if ref.exists():
        sys.exit("results/s3a2_hashes.json exists; already registered")
    ref.write_text(json.dumps({"files": cur, "registered": datetime.datetime.now().isoformat(timespec="minutes")}, indent=2))
    print(json.dumps(cur, indent=2))


if __name__ == "__main__":
    main()
