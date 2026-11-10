"""Amendment 6 splits (PREREGISTRATION_A6 §1). Built from generator_a4's task constructor, unchanged.

  a6_demo        SFT demonstration pool, DISJOINT from a4_train_A/B and every eval split (fresh seed stream).
                 Regime-A faults only (issue_refund): 60 UR (persistent phantom) + 20 RP (phantom, k_max = 2).
                 Demonstrations verify the refund only, so audit/close verification is never demonstrated (H12).
  e6_twin_rp     Recoverable twin of every e_unrecoverable task: identical world, ids, prompt and faulted tool,
                 but the phantom persists for k_max = 2 calls and then lands. A class recogniser that reports
                 "failed" on e_unrecoverable also fails here; an evidence-conditioned policy completes it (C2).
  e6_depth3      Phantom with k_max = 3 on seen tools: one more failure than any training task (C4, unseen depth).

    python -m actionproof.bench.generator_a6 --out data/bench/a6
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from actionproof.bench.generator import Task
from actionproof.bench.generator_a4 import EVAL_SPLITS, SEEN_WRITES, _task
from actionproof.env.faults import Fault
from actionproof.env.tools_a4 import HELD_OUT_TOOL

DEMO_SEED, N_DEMO_UR, N_DEMO_RP = 31, 60, 20
DEPTH3_SEED, N_DEPTH3 = 32, 150
SPLITS = ("a6_demo", "e6_twin_rp", "e6_depth3")


def make_demo_pool() -> list[Task]:
    out = []
    for i in range(N_DEMO_UR + N_DEMO_RP):
        rng = random.Random(f"actionproof-a6/demo/{DEMO_SEED}/{i}")
        if i < N_DEMO_UR:
            out.append(_task("a6_demo", i, rng, "refund", False, "UR", "issue_refund", Fault.PHANTOM, None, "A", False))
        else:
            out.append(_task("a6_demo", i, rng, "refund", False, "RP", "issue_refund", Fault.PHANTOM, 2, "A", False))
    return out


def make_twin_rp() -> list[Task]:
    """Replays e_unrecoverable's random stream draw for draw, changing only persistence (UR -> RP, k_max = 2)."""
    c = EVAL_SPLITS["e_unrecoverable"]
    out = []
    for i in range(c["n_ur"]):
        rng = random.Random(f"actionproof-a4/e_unrecoverable/{c['seed']}/{i}")
        group = i // 50
        tool = ("issue_refund", rng.choice(("add_audit_note", "close_ticket")), HELD_OUT_TOOL)[group]
        wf = "credit" if group == 2 else "refund"
        out.append(_task("e6_twin_rp", i, rng, wf, False, "RP", tool, Fault.PHANTOM, 2, None, True))
    return out


def make_depth3() -> list[Task]:
    out = []
    for i in range(N_DEPTH3):
        rng = random.Random(f"actionproof-a6/depth3/{DEPTH3_SEED}/{i}")
        tool = SEEN_WRITES[i % len(SEEN_WRITES)]
        out.append(_task("e6_depth3", i, rng, "refund", False, "RP", tool, Fault.PHANTOM, 3, None, True))
    return out


def make_a6_split(split: str) -> list[Task]:
    return {"a6_demo": make_demo_pool, "e6_twin_rp": make_twin_rp, "e6_depth3": make_depth3}[split]()


def split_bytes(split: str) -> bytes:
    return "".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in make_a6_split(split)).encode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench/a6")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (out / f"{s}.jsonl").write_bytes(split_bytes(s))
        print(s, len(make_a6_split(s)))


if __name__ == "__main__":
    main()
