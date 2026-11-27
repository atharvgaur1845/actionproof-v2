"""SEL1 SFT data for the two arms (PREREGISTRATION_SEL1.md). Built once; refuses to overwrite.

Both arms continue from the same parent (P4's Dw_aug, seed k, merged) with a new LoRA, one epoch, and differ only in
which TXN demonstrations they see:

- **SEL**: B5 episodes on every `sel1_train` task (core classes; quadruples in the standard, unresolvable and
  transient-1 regimes).
- **CTL**: B5 episodes on the *resolvable* `sel1_train` tasks (core classes and standard quadruples, the same
  episodes SEL gets), plus B5 episodes on `sel1_ctl_pool` (resolvable only) taken in file order until CTL has at least
  as many TXN rows as SEL. On resolvable tasks B5 behaves exactly as B4, so CTL never sees an outage or an
  escalation.
- **Replay** (both arms, identical): REPLAY_N rows drawn with a fixed seed from `data/rft/p4_aug_rows.jsonl` (the
  parents' own SFT data, P4-registered hash 3f3f99c8...).

Every episode is rendered exactly as `scripts/p1_eval.py` renders a model episode: its `start()` (system prompt with
the proc3 tool schemas, the task prompt), each agent call as `act_text` (assistant turn) and each observation through
`tool_text` (user turn). One row per assistant turn: prompt = the conversation before it, completion = that turn.
Only episodes the evaluator scores as safe terminal resolution (`txn.str`) are kept.

    python -m actionproof.sel1.build
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = {"SEL": ROOT / "data/rft/sel1_sel_rows.jsonl", "CTL": ROOT / "data/rft/sel1_ctl_rows.jsonl"}
STATS = ROOT / "data/rft/sel1_rows.stats.json"
REPLAY = ROOT / "data/rft/p4_aug_rows.jsonl"
REPLAY_SHA = "3f3f99c8d49d9cafc4987ce061702d5e74da364e92e4fc35d6f251428e99b051"
REPLAY_N = 2000
REPLAY_SEED = 20261127


def _p1():
    spec = importlib.util.spec_from_file_location("p1_eval", ROOT / "scripts/p1_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def render(task, env, P1) -> list[dict]:
    """The conversation a model would have produced, from the policy's env trace (agent calls only)."""
    from actionproof.stage3.rollout import act_text
    from actionproof.stage4.progress import request_total
    from actionproof.sel1.env import make_env_sel
    conv = P1.start(make_env_sel(task, f"sel1render-{task.task_id}"), task, False)
    total = request_total(task.prompt)
    for s in env.trace:
        if s.get("by", "agent") != "agent":
            continue
        conv.append({"role": "assistant", "content": act_text(s["tool"], s["args"])})
        conv.append({"role": "user", "content": P1.tool_text(s["tool"], s["obs"], total, False)})
    return conv


def rows_of(task, conv, arm: str) -> list[dict]:
    tx = task.meta["txn"]
    return [{"prompt": conv[:j], "completion": [conv[j]], "task_id": task.task_id, "arm": arm,
             "regime": tx["regime"], "group": tx["group"], "fault_class": task.meta.get("fault_class"),
             "policy": "B5", "turn": j} for j in range(2, len(conv)) if conv[j]["role"] == "assistant"]


def episodes(split: str, P1, stats: Counter) -> list[tuple]:
    from actionproof.sel1 import tasks as ST
    from actionproof.sel1.policy import run_policy
    out = []
    for t in ST.load(split):
        rec, env = run_policy("B5", t)
        if not rec["txn"]["str"] or rec["ended_by"] != "report":
            stats[f"{split}:dropped:{t.meta['txn']['regime']}"] += 1
            continue
        stats[f"{split}:kept:{t.meta['txn']['regime']}"] += 1
        stats[f"{split}:escalated"] += int(rec["txn"]["escalated"])
        out.append((t, render(t, env, P1)))
    return out


def main() -> None:
    clash = [str(p) for p in [*OUT.values(), STATS] if p.exists()]
    if clash:
        sys.exit(f"refusing to overwrite: {clash}")
    assert hashlib.sha256(REPLAY.read_bytes()).hexdigest() == REPLAY_SHA, "replay rows differ from P4's registered file"
    P1 = _p1()
    stats: Counter = Counter()
    train = episodes("sel1_train", P1, stats)
    pool = episodes("sel1_ctl_pool", P1, stats)
    sel = [r for t, c in train for r in rows_of(t, c, "SEL")]
    ctl = [r for t, c in train if t.meta["txn"]["regime"] in ("standard", "delayed") for r in rows_of(t, c, "CTL")]
    n_shared = len(ctl)
    used_pool = 0
    for t, c in pool:
        if len(ctl) >= len(sel):
            break
        ctl += rows_of(t, c, "CTL")
        used_pool += 1
    assert len(ctl) >= len(sel), f"control pool too small: {len(ctl)} < {len(sel)}"
    replay = [json.loads(l) for l in REPLAY.read_text().splitlines() if l.strip()]
    replay = random.Random(REPLAY_SEED).sample(replay, REPLAY_N)
    for r in replay:
        r["arm"] = "replay"
    for arm, txn_rows in (("SEL", sel), ("CTL", ctl)):
        OUT[arm].write_text("".join(json.dumps(r) + "\n" for r in replay + txn_rows))
    regimes = Counter(r["regime"] for r in sel)
    stats.update({"SEL_txn_rows": len(sel), "CTL_txn_rows": len(ctl), "CTL_rows_shared_with_SEL": n_shared,
                  "CTL_pool_episodes_used": used_pool, "replay_rows": REPLAY_N,
                  "SEL_escalate_turns": sum('"escalate_to_human"' in r["completion"][0]["content"] for r in sel),
                  "CTL_escalate_turns": sum('"escalate_to_human"' in r["completion"][0]["content"] for r in ctl),
                  **{f"SEL_rows_regime:{k}": v for k, v in regimes.items()}})
    for arm, p in OUT.items():
        stats[f"sha256:{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    STATS.write_text(json.dumps(dict(stats), indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(stats), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
