"""V2 training entry point: the first-stage trainer (actionproof/ad/train.py, unchanged) with a group-generation
condition and, optionally, a v2 advantage hook.

    python -m actionproof.ad2.train --groups resampled --estimator grpo --pools data/bench/proc/cw_train.jsonl \
        --seed 0 --init-adapter runs/rft_proc/final --out runs/ad2_grpo_rs_s0 --use-vllm

--groups shared     condition A (first-stage behaviour: one pool world per task)
--groups resampled  condition B (actionproof/ad2/rollout_rs.py: an i.i.d. prior world per rollout)
All other flags are passed to actionproof.ad.train unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    argv = sys.argv[1:]
    groups = "shared"
    if "--groups" in argv:
        i = argv.index("--groups")
        groups = argv[i + 1]
        del argv[i:i + 2]
    if groups not in ("shared", "resampled"):
        sys.exit(f"unknown --groups {groups}")
    if groups == "resampled":
        from actionproof.ad2 import rollout_rs
        rollout_rs.install()
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "ad2_config.json").write_text(json.dumps({"groups": groups, "argv": argv}, indent=2))
    sys.argv = [sys.argv[0], *argv]
    from actionproof.ad import train
    train.main()


if __name__ == "__main__":
    main()
