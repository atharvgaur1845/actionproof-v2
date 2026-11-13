"""Selection differential at step 0 (CPU; exploratory; Fable 5.2 next-steps review, analysis 1).

Question: does the warm start's own sample already predict which template GRPO keeps, and does the answer flip with
the reward's information channel?

Input: a warm start sampled T=1 x 4 on 100 pool-B tasks (the gate file), i.e. 100 groups like GRPO's first step.
Each episode is rescored under every arm's reward. Within each group, GRPO advantages are (r - mean) / std. For a
binary episode marker we report:
  * mean advantage of marker episodes minus non-marker episodes, over groups that contain both (paired within
    task), with a task-level bootstrap 95% CI (10k resamples);
  * the share of groups where the marker's mean advantage is positive.
A positive differential means the arm's first gradient step pushes the marked template up.

Markers (episodes with at least one `issue_refund` call):
  * demo_key: refund idempotency key == `refund-<order>` (the A6 demonstrator's template; H7-3 marker)
  * verifies: a `query_ledger` call after the first refund ack (post-write verification)

Caveats, stated in the output: the gate ran in the evaluation schema (`EcomEnvA4`, D-EVAL-1) while GRPO samples the
training schema; groups of 4, not 8; one sample of 100 tasks. This is a prediction from step-0 data, not a
measurement of training dynamics.

    python scripts/a7_selection_differential.py results/raw/a6/gate_init__a4_train_B_T1x4.jsonl \
        results/raw/a7/gate_init__a4_train_B_T1x4.jsonl > results/a7/selection_differential.md
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from types import SimpleNamespace

from actionproof.bench.generator_a4 import load_tasks
from actionproof.rewards import a6_rewards as A6
from actionproof.rewards.a7_rewards import pcstr_d_noFunder_score
from actionproof.rewards.ack_outcome import ack_outcome_score
from actionproof.rewards.baselines import tool_trust_score
from actionproof.rewards.pcstr import pcstr_score

GOALS = {t.task_id: t.goal for t in load_tasks("data/bench/a4/a4_train_B.jsonl")}


def _v(r):
    return SimpleNamespace(**r["verdict"])


ARMS = {  # name -> (channel, reward(rec))
    "M1 pcstr_d": ("state", lambda r: pcstr_score(_v(r), disclosure=True)),
    "M2 noreport": ("state", lambda r: A6.pcstr_d_noreport_score(_v(r))),
    "M2b noND": ("state", lambda r: A6.pcstr_d_noND_score(_v(r))),
    "M2c noFunder": ("state", lambda r: pcstr_d_noFunder_score(_v(r))),
    "M3 outcome": ("state", lambda r: float(r["verdict"]["P"] == 1 and r["verdict"]["I"] == 0)),
    "M3' outcome_G": ("state", lambda r: A6.outcome_G_score(_v(r))),
    "M4 ack_outcome": ("ack", lambda r: ack_outcome_score(GOALS[r["task_id"]], _model_trace(r), r["verdict"]["I"])),
    "M5 tool_trust": ("ack", lambda r: tool_trust_score(GOALS[r["task_id"]], _model_trace(r), r.get("report"))),
}


def _model_trace(r):
    return [s for s in r["trace"] if isinstance(s.get("args"), dict)]


def _args(s):
    return s.get("args") if isinstance(s.get("args"), dict) else {}


def markers(r) -> dict[str, bool] | None:
    t = [s for s in r["trace"] if not s.get("by")]
    refunds = [i for i, s in enumerate(t) if s["tool"] == "issue_refund"]
    if not refunds:
        return None
    order = _args(t[refunds[0]]).get("order_id")
    return {"demo_key": _args(t[refunds[0]]).get("idempotency_key") == f"refund-{order}",
            "verifies": any(s["tool"] == "query_ledger" for s in t[refunds[0] + 1:])}


def analyse(path: str, rng: random.Random) -> list[str]:
    recs = [json.loads(l) for l in open(path)]
    groups = defaultdict(list)
    for r in recs:
        groups[r["task_id"]].append(r)
    out = [f"### `{path}`", "",
           f"{len(groups)} groups x {len(recs) // len(groups)} samples. Episodes with a refund call: "
           f"{sum(markers(r) is not None for r in recs)}."]
    for mk in ("demo_key", "verifies"):
        n_mk = sum(bool(m and m[mk]) for m in map(markers, recs))
        out += ["", f"**Marker `{mk}`** ({n_mk} marked episodes)", "",
                "| arm | channel | mixed groups | mean adv(marked) − adv(unmarked) | 95% CI (task bootstrap) |"
                " groups with positive differential |", "|---|---|---|---|---|---|"]
        for arm, (ch, fn) in ARMS.items():
            diffs = []
            for g in groups.values():
                ms = [markers(r) for r in g]
                if any(m is None for m in ms):
                    continue
                lab = [m[mk] for m in ms]
                if all(lab) or not any(lab):
                    continue
                rs = [fn(r) for r in g]
                sd = statistics.pstdev(rs)
                if sd < 1e-9:
                    diffs.append(0.0)
                    continue
                adv = [(x - statistics.mean(rs)) / sd for x in rs]
                a1 = [x for x, l in zip(adv, lab) if l]
                a0 = [x for x, l in zip(adv, lab) if not l]
                diffs.append(statistics.mean(a1) - statistics.mean(a0))
            if not diffs:
                out.append(f"| {arm} | {ch} | 0 | – | – | – |")
                continue
            boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
            out.append(f"| {arm} | {ch} | {len(diffs)} | {statistics.mean(diffs):+.3f} | "
                       f"[{boots[249]:+.3f}, {boots[9749]:+.3f}] | {sum(d > 0 for d in diffs)}/{len(diffs)} |")
    return out


def main() -> None:
    rng = random.Random(0)
    lines = ["# Step-0 selection differential (exploratory, CPU)", "",
             "Generated by `scripts/a7_selection_differential.py`. Caveats: the gate files were sampled in the "
             "evaluation schema (D-EVAL-1), groups of 4 (training uses 8), 100 tasks. Groups with zero reward "
             "variance contribute a differential of 0 (GRPO gives them no gradient).", ""]
    for p in sys.argv[1:]:
        lines += analyse(p, rng) + [""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
