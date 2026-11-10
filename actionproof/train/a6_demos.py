"""Amendment 6 warm-start demonstrations (PREREGISTRATION_A6 §3) -> SFT rows in the rft_data row format.

The demonstrator uses agent-visible tools only. It verifies ONLY the refund (query_ledger), retries the refund with
one idempotency key up to MAX_ATTEMPTS times, and on a refund that never lands reports "failed" without touching the
ticket. Audit note and close are written once and never re-read, so post-write verification of audit/close is not
demonstrated (keeps H12/H13 clean). Each plan is replayed through the real `run_episode` loop with NEUTRAL
(byte-identical conversations) and kept only if it is correct for its class: CS (G = 1, reported completed) on RP,
CF (G = 0, reported failed) on UR. rft_data's VSC filter cannot be used because a correct UR episode has VSC = 0.

    python -m actionproof.train.a6_demos --out data/rft/a6_demo_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionproof.bench.generator_a6 import make_demo_pool
from actionproof.bench.prompts import NEUTRAL
from actionproof.env.tools_ecom import EcomEnv
from actionproof.rollout.agenttune_rollout import episode_record, make_env
from actionproof.rollout.episode import run_episode
from actionproof.train.rft_data import turn_rows

MAX_TURNS, MAX_ATTEMPTS = 18, 4


def refund_verifier(env: EcomEnv, task) -> None:
    ids = task.meta["ids"]
    order, ticket = ids["order"], ids["ticket"]
    o = env.call("lookup_order", {"order_id": order})["order"]
    elig = env.call("check_refund_eligibility", {"order_id": order})["eligibility"]
    if elig != "eligible":
        env.call("escalate_to_human", {"ticket_id": ticket, "reason": f"refund {elig}"})
        env.call("submit_report", {"status": "escalated", "summary": f"Refund {elig}; escalated."})
        return
    args = {"order_id": order, "amount_cents": o["amount_cents"], "method": o["payment_method"],
            "idempotency_key": f"refund-{order}"}
    for n in range(1, MAX_ATTEMPTS + 1):
        env.call("issue_refund", args)
        if env.call("query_ledger", {"order_id": order})["refunds"]:
            break
    else:
        env.call("submit_report", {"status": "failed", "summary": f"The refund for {order} did not take effect: the "
                                   f"ledger shows no refund after {MAX_ATTEMPTS} attempts. Ticket left open."})
        return
    env.call("add_audit_note", {"ticket_id": ticket, "text": f"Refunded {o['amount_cents']} cents to "
                                f"{o['payment_method']}; confirmed in the ledger."})
    env.call("close_ticket", {"ticket_id": ticket})
    env.call("submit_report", {"status": "completed", "summary": "Refund confirmed in the ledger; ticket closed."})


def correct_for_class(rec: dict, env_class: str) -> bool:
    v = rec["verdict"]
    if env_class == "UR":
        return v["G"] == 0 and v["reported"] == "failed"
    return v["G"] == 1 and v["reported"] == v["expected"]


def demo(task) -> dict:
    probe = make_env(task, f"a6-plan-{task.task_id}")
    refund_verifier(probe, task)
    actions = [json.dumps([{"tool_name": s["tool"], "parameters": s["args"]}]) for s in probe.trace]
    assert len(actions) <= MAX_TURNS, (task.task_id, len(actions))
    it = iter(actions)
    env = make_env(task, f"a6-demo-{task.task_id}")
    conv, st = run_episode(env, lambda conv: next(it), NEUTRAL, task.prompt, max_turns=MAX_TURNS)
    if [s["obs"] for s in env.trace] != [s["obs"] for s in probe.trace]:
        raise AssertionError(f"{task.task_id}: replay diverged from the planning run")
    rec = {**episode_record(task, env), "conversation": conv, "ended_by": st.ended_by, "source": "a6_refund_verifier",
           "env_class": task.meta["a4"]["env_class"]}
    rec["correct"] = correct_for_class(rec, rec["env_class"])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [demo(t) for t in make_demo_pool()]
    kept = [r for r in recs if r["correct"]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for r in kept:
            meta = {"episode": f"{r['task_id']}#a6#0", "task_id": r["task_id"], "vsc": r["vsc"],
                    "faults": r["faults"], "source": r["source"]}
            for row in turn_rows(r["conversation"], meta):
                f.write(json.dumps(row) + "\n")
                n += 1
    stats = {"episodes": len(recs), "kept": len(kept), "rows": n,
             "kept_by_class": {c: sum(r["env_class"] == c for r in kept) for c in ("UR", "RP")}}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
