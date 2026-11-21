"""S3 addendum 3 secondary: the frozen S3 evaluation (`scripts/s3_eval.py`, not edited) with each recovery start's
scripted prefix replayed into the environment but not into the conversation (as `train_variants.py --hide-prefix`).
    python scripts/s3_eval_hidden.py --model runs/rft_proc/merged --adapter runs/X/final --tag X --split s3_conf_rec \
        --out-dir results/s3/hidden
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import actionproof.stage3.rollout as R
from actionproof.stage4.train_variants import hide_prefix

ROOT = Path(__file__).resolve().parents[1]
hide_prefix(R)
_spec = importlib.util.spec_from_file_location("s3_eval", ROOT / "scripts/s3_eval.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
E.start_conversation = R.start_conversation

if __name__ == "__main__":
    E.main()
