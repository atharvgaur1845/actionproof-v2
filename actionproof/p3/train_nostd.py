"""P3 arm Dn (PREREGISTRATION_P3.md): P1 arm D's training (`actionproof/stage3/train.py`, unchanged) with TRL's
scale_rewards = "none". Advantages are centred within the group and not divided by the group's reward standard
deviation, which is the Dr. GRPO estimator confirmed by AD-C1. Everything else is P1 arm D's: warm start, pools,
reward R_V, steps, seeds, LoRA, learning rate, KL and group size.

    python -m actionproof.p3.train_nostd --reward rv --pools ... --init-adapter runs/p1_dwarm/final --out runs/p3_dn_s0 --seed 0 --use-vllm
"""
from __future__ import annotations

import agenttune.backends.trl.agentic.grpo.agentic_grpo as ag

from actionproof.stage3 import train as s3t

_Orig = ag.TrlAgenticGrpo


class _NoStd(_Orig):
    def __init__(self, *args, **kwargs):
        kwargs["scale_rewards"] = "none"
        super().__init__(*args, **kwargs)


def main() -> None:
    ag.TrlAgenticGrpo = _NoStd  # stage3.train imports the class from this module inside main()
    s3t.main()


if __name__ == "__main__":
    main()
