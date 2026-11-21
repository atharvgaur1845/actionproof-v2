"""Stage 4 shift evaluation (PREREGISTRATION_S4_EVAL): the frozen S3 loop (`scripts/s3_eval.py`, imported unchanged)
with the API-shifted environment for `s4_api` and the canonical environment for every other split.
    python scripts/s4_eval.py --model runs/rft_proc/merged --adapter runs/a12_proc_rv_s0/final --tag a12_rv_s0 --split s4_api
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s3_eval", ROOT / "scripts/s3_eval.py")
s3e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3e)

if __name__ == "__main__":
    split = sys.argv[sys.argv.index("--split") + 1]
    if split == "s4_api":
        from actionproof.stage4.shift import make_env_api
        s3e.make_env_s3 = make_env_api
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", "results/s4/eval"]
    s3e.main()
