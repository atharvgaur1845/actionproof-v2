"""Amendment 7 credit-workflow demonstrations (PREREGISTRATION_A7 §3) -> SFT rows in the rft_data row format.

The demonstrator is the A6 refund verifier's template, extended by one step: lookup -> eligibility -> refund with
key `refund-<order>`, re-read the ledger, retry up to MAX_ATTEMPTS -> `apply_store_credit` ONCE with its own key
`credit-<order>` (never the refund's: see D-ENV-1) -> audit note -> close -> report. The credit is not re-read, so
credit verification is never demonstrated. Plans are replayed through `run_episode` in EcomEnvA4 (the only env with
the credit tool) and kept only if G = 1, the report is "completed" and an `apply_store_credit` receipt exists.

    python -m actionproof.train.a7_demos --out data/rft/a7_credit_demo_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionproof.bench.generator_a7 import make_credit_demo_pool
from actionproof.bench.prompts import NEUTRAL
from actionproof.env.tools_ecom import EcomEnv
from actionproof.eval.a4_runtime import episode_record_a4, make_env_a4
from actionproof.rollout.episode import run_episode
from actionproof.train.a6_demos import MAX_ATTEMPTS, MAX_TURNS
from actionproof.train.rft_data import turn_rows


def credit_verifier(env: EcomEnv, task) -> None:
    ids = task.meta["ids"]
    order, ticket, credit = ids["order"], ids["ticket"], ids["credit"]
    o = env.call("lookup_order", {"order_id": order})["order"]
    elig = env.call("check_refund_eligibility", {"order_id": order})["eligibility"]
    if elig != "eligible":
        env.call("escalate_to_human", {"ticket_id": ticket, "reason": f"refund {elig}"})
        env.call("submit_report", {"status": "escalated", "summary": f"Refund {elig}; escalated."})
        return
    args = {"order_id": order, "amount_cents": o["amount_cents"], "method": o["payment_method"],
            "idempotency_key": f"refund-{order}"}
    for _ in range(MAX_ATTEMPTS):
        env.call("issue_refund", args)
        if env.call("query_ledger", {"order_id": order})["refunds"]:
            break
    else:
        env.call("submit_report", {"status": "failed", "summary": f"The refund for {order} did not take effect: the "
                                   f"ledger shows no refund after {MAX_ATTEMPTS} attempts. Ticket left open."})
        return
    env.call("apply_store_credit", {"order_id": order, "amount_cents": credit, "idempotency_key": f"credit-{order}"})
    env.call("add_audit_note", {"ticket_id": ticket, "text": f"Refunded {o['amount_cents']} cents to "
                                f"{o['payment_method']}; confirmed in the ledger. Applied a goodwill store credit of "
                                f"{credit} cents."})
    env.call("close_ticket", {"ticket_id": ticket})
    env.call("submit_report", {"status": "completed",
                               "summary": "Refund confirmed in the ledger; store credit applied; ticket closed."})


def correct(rec: dict) -> bool:
    v = rec["verdict"]
    return v["G"] == 1 and v["reported"] == "completed" and any(r["op"] == "apply_store_credit" for r in rec["receipts"])


def demo(task) -> dict:
    probe = make_env_a4(task, f"a7-plan-{task.task_id}")
    credit_verifier(probe, task)
    actions = [json.dumps([{"tool_name": s["tool"], "parameters": s["args"]}]) for s in probe.trace]
    assert len(actions) <= MAX_TURNS, (task.task_id, len(actions))
    it = iter(actions)
    env = make_env_a4(task, f"a7-demo-{task.task_id}")
    conv, st = run_episode(env, lambda conv: next(it), NEUTRAL, task.prompt, max_turns=MAX_TURNS)
    if [s["obs"] for s in env.trace] != [s["obs"] for s in probe.trace]:
        raise AssertionError(f"{task.task_id}: replay diverged from the planning run")
    rec = {**episode_record_a4(task, env), "conversation": conv, "ended_by": st.ended_by,
           "source": "a7_credit_verifier"}
    rec["correct"] = correct(rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [demo(t) for t in make_credit_demo_pool()]
    kept = [r for r in recs if r["correct"]]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for r in kept:
            meta = {"episode": f"{r['task_id']}#a7#0", "task_id": r["task_id"], "vsc": r["vsc"],
                    "faults": r["faults"], "source": r["source"]}
            for row in turn_rows(r["conversation"], meta):
                f.write(json.dumps(row) + "\n")
                n += 1
    stats = {"episodes": len(recs), "kept": len(kept), "rows": n}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
