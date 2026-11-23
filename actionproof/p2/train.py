"""Arm E's GRPO training (PREREGISTRATION_P2.md §4): `actionproof/stage3/train.py` run unchanged, with its rollout
replaced by the ledger rollout (`actionproof/p2/rollout.py`). Same arguments as stage3.train.

    python -m actionproof.p2.train --reward rv --pools ... --init-adapter runs/p2_ewarm/final --out runs/p2_e_s0 --use-vllm
"""
from __future__ import annotations

import actionproof.p2.rollout as p2r
import actionproof.stage3.rollout as s3r
from actionproof.stage3 import train as s3t


def main() -> None:
    s3r.make_rollout_func = p2r.make_rollout_func  # stage3.train imports it from this module inside main()
    s3t.main()


if __name__ == "__main__":
    main()
