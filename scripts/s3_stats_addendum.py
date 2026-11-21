"""S3 addendum 1 statistics (PREREGISTRATION_S3_ADDENDUM1 §3): `scripts/s3_stats.py` (frozen, imported unchanged) with
the comparison RS-GRPO vs the class-matched baseline, optionally restricted to fault classes of one split.
    python scripts/s3_stats_addendum.py --cand s3_rv_mix --base s3_matched --seeds 0,1,2 --out results/s3/s3_addendum.json
    python scripts/s3_stats_addendum.py ... --classes partial,dup_delivery --out results/s3/s3_addendum_full_pd.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s3_stats", ROOT / "scripts/s3_stats.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--split", default=None)
    a, rest = ap.parse_known_args()
    if a.split:
        s3.SPLITS = (a.split,)
    if a.classes:
        keep = set(a.classes.split(","))
        orig = s3.load

        def load(tag, split, d):
            p = ROOT / d / f"{tag}__{split}.jsonl"
            if not p.exists():
                return None
            ids = {r["task_id"] for r in map(json.loads, p.open()) if r["fault_class"] in keep}
            full = orig(tag, split, d)
            return {k: v for k, v in full.items() if k in ids}
        s3.load = load
    sys.argv = [sys.argv[0]] + rest
    s3.main()
