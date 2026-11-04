"""Episode rendering for the demo (no Gradio import, so scripts can build cached replays)."""
from __future__ import annotations

import json

from actionproof.env.db import snapshot
from actionproof.env.tools_ecom import VERIFICATION_TOOLS
from actionproof.env.verifier import verify, vsc
from actionproof.rewards.baselines import outcome_score, tool_trust_score
from actionproof.rewards.pcstr import pcstr_score


def score(task, env) -> dict:
    v = verify(task.goal, snapshot(env.conn), env.log, env.final_report)
    return {"verdict": v, "VSC": vsc(v),
            "rewards": {"tool-trusting (transcript)": tool_trust_score(task.goal, env.trace, env.final_report),
                        "outcome (DB)": outcome_score(v), "PCSTR-d (ours)": round(pcstr_score(v, disclosure=True), 3)},
            "n_verify": sum(s["tool"] in VERIFICATION_TOOLS for s in env.trace)}


def render(task, env, s) -> str:
    att = {a.action_idx: a for a in env.log.attempts}
    lines = []
    for st in env.trace:
        args = ", ".join(f"{k}={v}" for k, v in st["args"].items() if k != "summary")
        line = f"**{st['tool']}**({args}) → `{json.dumps(st['obs'])[:140]}`"
        a = att.get(st["idx"])
        if a is not None:
            truth = "written ✅" if a.committed and a.fault != "F3_partial" else (
                "PARTIALLY written ⚠️" if a.fault == "F3_partial" else "NOT written ❌")
            line += f"<br>&nbsp;&nbsp;&nbsp;↳ *database:* **{truth}**" + (f" (`{a.fault}`)" if a.fault else "")
        lines.append(line)
    snap = snapshot(env.conn)
    o = task.meta["ids"]["order"]
    db = {"refunds for order": [r for r in snap["refunds"] if r["order_id"] == o],
          "order status": next(r["status"] for r in snap["orders"] if r["order_id"] == o),
          "ticket": next({k: r[k] for k in ("status",)} for r in snap["tickets"] if r["order_id"] == o),
          "escalations": len(snap["escalations"])}
    v = s["verdict"]
    report = env.final_report or {"status": "(no report)"}
    verdict = ("✅ **Verified Safe Completion**" if s["VSC"] else
               "❌ **False completion**: claimed done, the database disagrees" if v.F_over else "⚠️ Not completed")
    rw = " | ".join(f"{k}: **{val}**" for k, val in s["rewards"].items())
    return (f"#### Agent said: `{report['status']}`. {verdict}\n\n" + "<br>".join(lines) +
            f"\n\n**Actual database state:** `{json.dumps(db)}`\n\n**Rewards for this episode:** {rw}"
            f"\n\nverification calls: {s['n_verify']} · duplicates: {v.D} · violations: {v.I:.2f}")
