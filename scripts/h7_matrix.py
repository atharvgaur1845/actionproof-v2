"""A11 H7: which recovery defences survive which execution faults, on identical fault traces (scripted, CPU).

Worlds: the 200 `proc_dev` tasks (never a confirmatory split). For each world and each requested step, every fault
kind below is injected on that step's first write; every defence then runs on exactly that (world, fault plan).

Defences (scripted policies, agent-visible tools only):
  D0 trust            write once; never read; report completed
  D1 keys+retry       retry a timed-out write with the same idempotency key (no reads)
  D2 retry new key    retry a timed-out write with a fresh key (no idempotency protection)
  D3 read-back only   write, read the record, report from what it shows (no retry)
  D4 verify-retry     read; if absent, retry with the same key (at most 3 writes); report from the evidence
  D5 reconcile        D4 plus repair (refund extra or partial payment and re-pay; record missing receipt quantity)
Metrics per (defence, fault): goal met, duplicate irreversible effect, false finalization (report completed while the
goal is not met), honest disclosure (report failed when the goal is not met), raw residual exposure in cents (money
paid beyond the PO amount, or paid without a verified payment step), calls.

    python scripts/h7_matrix.py --out results/a11/h7_matrix.md
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from actionproof.bench.generator_proc import load_tasks
from actionproof.integrity.sim.agents import _args, _observed, _po
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc

ROOT = Path(__file__).resolve().parents[1]
FAULTS = ("phantom", "partial", "lost_ack", "precommit_timeout", "dup_delivery", "never_commit")
DEFENCES = ("D0 trust", "D1 keys+retry", "D2 retry new key", "D3 read-back only", "D4 verify-retry", "D5 reconcile")


def run_defence(d: str, env, task) -> None:
    po = _po(task)
    pid = po["po_id"]
    failed = False
    for step in task.meta["requested"]:
        key = f"{step}-{pid}"
        if d in ("D0 trust", "D1 keys+retry", "D2 retry new key"):
            for n in range(1 if d == "D0 trust" else 3):
                k = key if d != "D2 retry new key" else f"{key}-{n}"
                if env.call(STEP_TOOL[step], _args(step, po, k)).get("status") == "ok":
                    break
            continue
        held = False
        for _ in range(1 if d == "D3 read-back only" else 3):
            env.call(STEP_TOOL[step], _args(step, po, key))
            seen = _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po)
            if seen == "held":
                held = True
                break
            if d == "D5 reconcile" and step == "pay" and seen in ("partial", "other"):
                pays = env.call("list_payments", {"po_id": pid})["payments"]
                refunded = {p["refund_of"] for p in pays if p["kind"] == "refund"}
                live = [p for p in pays if p["kind"] == "payment" and p["payment_id"] not in refunded]
                for p in (live if seen == "partial" else live[1:]):
                    env.call("refund_payment", {"payment_id": p["payment_id"]})
                if seen == "partial":
                    key = f"{key}-r"
                    continue
                held = _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po) == "held"
                break
            if d == "D5 reconcile" and step == "receive" and seen == "partial":
                got = sum(r["qty"] for r in env.call("list_goods_receipts", {"po_id": pid})["receipts"])
                env.call(STEP_TOOL[step], {"po_id": pid, "qty": po["qty"] - got})
                held = _observed(step, env.call(STEP_READ[step], {"po_id": pid}), po) == "held"
                break
            if seen != "absent":
                break
        if not held:
            failed = True
            break
    env.call("submit_report", {"status": "failed" if failed else "completed", "summary": d})


def exposure(rec) -> int:
    """Cents paid to the supplier for the task PO beyond one verified payment of the PO amount (net of refunds)."""
    po = next(p for p in rec["task"]["initial_state"]["pos"] if p["po_id"] == rec["task"]["meta"]["po_id"])
    pays = rec["final_state"]["payments"]
    refunded = {p["refund_of"] for p in pays if p["kind"] == "refund"}
    init = {p["payment_id"] for p in rec["task"]["initial_state"]["payments"]}
    live = [p for p in pays if p["kind"] == "payment" and p["po_id"] == po["po_id"] and p["payment_id"] not in refunded
            and p["payment_id"] not in init]
    paid = sum(p["amount_cents"] for p in live)
    ok = rec["score"]["steps"].get("pay") == "VERIFIED_SUCCESS"
    return max(0, paid - po["amount_cents"]) if ok else paid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/a11/h7_matrix.md")
    a = ap.parse_args()
    tasks = load_tasks(ROOT / "data/bench/proc/proc_dev.jsonl")
    agg = defaultdict(lambda: defaultdict(float))
    for t0 in tasks:
        for step in t0.meta["requested"]:
            for f in FAULTS:
                t = copy.deepcopy(t0)
                if f == "never_commit":
                    t.meta.update(faults=[], never_commit=STEP_TOOL[step])
                else:
                    t.meta.update(faults=[{"tool": STEP_TOOL[step], "k": 0, "kind": f}], never_commit=None)
                for d in DEFENCES:
                    env = make_env_proc(t, "h7")
                    run_defence(d, env, t)
                    rec = episode_record_proc(t, env)
                    s, g = rec["score"], agg[(d, f)]
                    g["n"] += 1
                    g["goal"] += s["goal_met"]
                    g["dup"] += s["duplicate_irreversible"]
                    g["false_final"] += s["false_success"]
                    g["disclosed"] += (not s["goal_met"]) and env.final_report["status"] == "failed"
                    g["exposure"] += exposure(rec)
                    g["calls"] += len(env.trace)
    rows = []
    md = ["# H7: defences x execution faults on identical fault traces (scripted policies, ProcureSim proc_dev worlds)", "",
          "Generated by `scripts/h7_matrix.py`. Each cell: the same 200 worlds x every requested step with that fault on its "
          "first write. Rates are over episodes; exposure is mean cents paid beyond one verified PO payment.", "",
          "| defence | fault | n | goal met | duplicate | false finalization | honest disclosure | mean exposure (cents) | mean calls |",
          "|---|---|---|---|---|---|---|---|---|"]
    for f in FAULTS:
        for d in DEFENCES:
            g = agg[(d, f)]
            n = g["n"]
            row = {"defence": d, "fault": f, "n": int(n), **{k: g[k] / n for k in ("goal", "dup", "false_final", "disclosed",
                                                                                  "exposure", "calls")}}
            rows.append(row)
            md.append(f"| {d} | {f} | {int(n)} | {row['goal']:.3f} | {row['dup']:.3f} | {row['false_final']:.3f} | "
                      f"{row['disclosed']:.3f} | {row['exposure']:.0f} | {row['calls']:.1f} |")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
