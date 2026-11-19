"""Amendment 12 registration artefact: the reward-truth table (written before any training).

For one requested step (`pay`, a keyed, irreversible write) on a fixed task, every combination of
(fault class) x (agent behaviour) is played by a small scripted policy in the real environment and scored with
the real verifier. The table shows, per cell, R_V and R_A, the turns used, whether the policy read the
authoritative record, and VSC. From it we derive, per fault class and reward, the sign of verification's
relative advantage: the best reward a reading behaviour reaches minus the best a non-reading behaviour reaches
(a tie is "0, minus turn cost": the read consumes a turn of a 16-turn budget and earns nothing).

Nothing here is tuned: the rewards are exactly `rewards/proc_rewards.py`, the behaviours are fixed below.

    python scripts/a12_reward_truth.py --out results/a12/reward_truth_table.md
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from actionproof.bench.generator_proc import make_split
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc

FAULTS = {
    "clean": ([], None),
    "phantom (success ack, nothing stored)": ([{"tool": "pay_supplier", "k": 0, "kind": "phantom"}], None),
    "partial (success ack, half stored)": ([{"tool": "pay_supplier", "k": 0, "kind": "partial"}], None),
    "lost ack (stored, timeout ack)": ([{"tool": "pay_supplier", "k": 0, "kind": "lost_ack"}], None),
    "precommit timeout (not stored, timeout ack)": ([{"tool": "pay_supplier", "k": 0, "kind": "precommit_timeout"}], None),
    "never commits (success ack every time)": ([], "pay_supplier"),
    "duplicate delivery (stored twice, success ack)": ([{"tool": "pay_supplier", "k": 0, "kind": "dup_delivery"}], None),
}
READS = {"read then proceed", "read, retry same key if missing", "read, stop and report failed if missing",
         "read, escalate if missing", "read, repair (refund extras or partial, re-pay)"}


def _pay(env, po, key="k1"):
    return env.call("pay_supplier", {"po_id": po["po_id"], "amount_cents": po["amount_cents"], "idempotency_key": key})


def _state(env, po):
    obs = env.call("list_payments", {"po_id": po["po_id"]})
    refunded = {p["refund_of"] for p in obs["payments"] if p["kind"] == "refund"}
    return [p for p in obs["payments"] if p["kind"] == "payment" and p["payment_id"] not in refunded]


def behave(name: str, env, po) -> None:
    ack = _pay(env, po)
    if name == "proceed without reading":
        pass
    elif name == "blind retry same key on error ack":
        if ack["status"] != "ok":
            _pay(env, po)
    elif name == "blind retry new key on error ack":
        if ack["status"] != "ok":
            _pay(env, po, key="k2")
    elif name == "read then proceed":
        _state(env, po)
    elif name == "read, retry same key if missing":
        if not _state(env, po):
            _pay(env, po)
    elif name == "read, stop and report failed if missing":
        live = _state(env, po)
        if len(live) != 1 or live[0]["amount_cents"] != po["amount_cents"]:
            env.call("submit_report", {"status": "failed", "summary": "payment not confirmed"})
            return
    elif name == "read, escalate if missing":
        live = _state(env, po)
        if len(live) != 1 or live[0]["amount_cents"] != po["amount_cents"]:
            env.call("escalate_to_human", {"po_id": po["po_id"], "reason": "payment not confirmed"})
            env.call("submit_report", {"status": "escalated", "summary": "payment not confirmed"})
            return
    elif name == "read, repair (refund extras or partial, re-pay)":
        live = _state(env, po)
        if not live:
            _pay(env, po)
        elif len(live) > 1:
            for p in live[1:]:
                env.call("refund_payment", {"payment_id": p["payment_id"]})
        elif live[0]["amount_cents"] != po["amount_cents"]:
            env.call("refund_payment", {"payment_id": live[0]["payment_id"]})
            _pay(env, po, key="k2")
    else:
        raise ValueError(name)
    env.call("submit_report", {"status": "completed", "summary": "paid"})


BEHAVIOURS = ["proceed without reading", "blind retry same key on error ack", "blind retry new key on error ack",
              "read then proceed", "read, retry same key if missing", "read, stop and report failed if missing",
              "read, escalate if missing", "read, repair (refund extras or partial, re-pay)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/a12/reward_truth_table.md")
    a = ap.parse_args()
    base = next(t for t in make_split("proc_demo") if t.workflow == "pay_post")
    rows, signs = [], []
    for fname, (faults, never) in FAULTS.items():
        best = {"R_V": {True: 0, False: 0}, "R_A": {True: 0, False: 0}}
        for b in BEHAVIOURS:
            t = copy.deepcopy(base)
            t.meta.update(requested=["pay"], faults=faults, never_commit=never)
            po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
            env = make_env_proc(t, "truth")
            behave(b, env, po)
            s = episode_record_proc(t, env)["score"]
            reads = b in READS
            best["R_V"][reads] = max(best["R_V"][reads], s["r_v"])
            best["R_A"][reads] = max(best["R_A"][reads], s["r_a"])
            rows.append((fname, b, s["r_v"], s["r_a"], len(env.trace), "yes" if reads else "no", s["vsc"],
                         s["steps"]["pay"]))
        sign = {}
        for r in ("R_V", "R_A"):
            d = best[r][True] - best[r][False]
            sign[r] = "+" if d > 0 else "-" if d < 0 else "0 (minus turn cost)"
        signs.append((fname, sign["R_V"], sign["R_A"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = ["# A12 reward-truth table (registration artefact; computed before training)", "",
          f"Task: `{base.task_id}` reduced to one requested step (`pay`). Rewards and verifier are the registered "
          "code (`rewards/proc_rewards.py`, `actionproof/integrity/sim/truth.py`). Generated by "
          "`scripts/a12_reward_truth.py`.", "",
          "## Per cell", "", "| fault | behaviour | R_V | R_A | calls | reads? | VSC | pay verdict |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {f} | {b} | {rv} | {ra} | {n} | {rd} | {v} | {pv} |" for f, b, rv, ra, n, rd, v, pv in rows]
    md += ["", "## Sign of verification's relative advantage",
           "", "Best reward over reading behaviours minus best reward over non-reading behaviours, per fault class.",
           "", "| fault | R_V | R_A |", "|---|---|---|"]
    md += [f"| {f} | {a_} | {b_} |" for f, a_, b_ in signs]
    md += ["", "Reading this table: under R_A, reading can only help when it changes which acknowledgement comes "
           "last; where acting on what the read shows means *not* retrying (a lost ack), the read-informed policy "
           "ends on the timeout ack and R_A pays 0. Under R_V, reading is the only route to reward on phantom and "
           "partial writes."]
    out.write_text("\n".join(md) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps({"rows": rows, "signs": signs}, indent=1))
    print("\n".join(md[-(len(signs) + 6):]))


if __name__ == "__main__":
    main()
