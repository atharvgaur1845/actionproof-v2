"""Held-out-tool exclusion audit (PREREGISTRATION_A4 §4.3). Exit 1 on any hit.

Channels: (1) teacher / RFT / demo / curriculum data under data/ (every file except the A4 *evaluation* splits);
(2) training-pool reward specs; (3) training system prompts as GRPO renders them (EcomEnv + NEUTRAL);
(4) the init's training lineage (RFT run dirs); (5) GRPO run dirs passed with --runs (logs, args, completions);
(6) the training env itself (H not listed, unknown_tool if called).

    python scripts/a4_exclusion_audit.py [--runs runs/grpo_tiny_a4A_outcome_s0 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from actionproof.bench.generator_a4 import TRAIN_POOLS, load_tasks
from actionproof.bench.prompts import NEUTRAL
from actionproof.env.tools_a4 import HELD_OUT_TOOL
from actionproof.rollout.agenttune_rollout import make_env
from actionproof.rollout.episode import system_prompt

TOKEN = HELD_OUT_TOOL.encode()
TEXT = (".jsonl", ".json", ".txt", ".log", ".md", ".py", ".csv")


def grep(paths) -> list[str]:
    hits = []
    for p in paths:
        for f in ([p] if p.is_file() else p.rglob("*")):
            if f.is_file() and f.suffix in TEXT and f.stat().st_size < 2_000_000_000 and TOKEN in f.read_bytes():
                hits.append(str(f))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--init-runs", nargs="*", default=["runs/rft_tiny_v2"])
    a = ap.parse_args()
    report = {}
    data_files = [f for f in Path("data").rglob("*") if f.is_file()
                  and not (f.parent.name == "a4" and f.name.startswith("e_"))]
    report["1_data"] = grep(data_files)
    pools = [Path("data/bench/a4") / f"{s}.jsonl" for s in TRAIN_POOLS]
    tasks = [t for p in pools for t in load_tasks(str(p))]
    report["2_reward_specs"] = [t.task_id for t in tasks if any(op == HELD_OUT_TOOL for op, _ in t.goal.required_writes)
                                or any(pc["p"] == "credit" for pc in t.goal.postconditions)]
    report["3_training_prompts"] = sorted({t.task_id for t in tasks
                                           if HELD_OUT_TOOL in system_prompt(make_env(t, "audit"), NEUTRAL) + t.prompt})
    report["4_init_lineage"] = grep([Path(p) for p in a.init_runs if Path(p).exists()])
    report["5_grpo_runs"] = grep([Path(p) for p in a.runs if Path(p).exists()])
    env = make_env(tasks[0], "audit")
    report["6_training_env"] = ([] if HELD_OUT_TOOL not in [f.__name__ for f in env.tools()]
                                and env.call(HELD_OUT_TOOL, {})["error"] == "unknown_tool" else ["listed or callable"])
    bad = {k: v for k, v in report.items() if v}
    print(json.dumps({k: (len(v), v[:5]) for k, v in report.items()}, indent=1))
    print("EXCLUSION AUDIT:", "PASS (0 hits on every channel)" if not bad else f"FAIL {list(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
