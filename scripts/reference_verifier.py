"""Independent reference verifier for ActionProof (written from the spec only).

Implemented from docs/ARTICLE_DRAFT.md §2, PREREGISTRATION.md / PREREGISTRATION_A4.md §1.1
(metric definitions), actionproof/env/tools_ecom.py + faults.py (tool semantics) and
actionproof/bench/generator.py (task goal format). It does NOT import or read the project
verifier (actionproof/env/verifier*.py), the rewards package or eval/a4_metrics.py, and it
never reads the stored `verdict`, `violations` or `vsc` fields when computing.

Inputs per episode: `task_id`, `trace` (tool, args, obs per step) and `attempts`
(action_idx, op, entity, fault, committed). The task (initial DB, goal postconditions,
required writes, allowed entities, feasibility) is regenerated with make_split(split).

Method: replay the trace on a plain-dict copy of the task's initial DB. A write step changes
state only if an attempt with that action_idx exists and committed=True; an F3_partial attempt
applies only the first sub-effect (tools_ecom._write: deltas[:1]). Then
  G   = all postconditions hold on the final state AND no invariant violated AND no duplicate
        irreversible committed write AND every required (op, entity) has a committed attempt
  exp = "completed" if feasible else "escalated"; rep = first accepted submit_report status
  VSC = 1[G = 1 and rep == exp].
Invariants (3): policy (every committed refund was eligible at call time, 0 < amount <= order
amount, method == order payment method), refund total per order <= order amount, scope (every
committed write targets an allowed entity). Duplicate: > 1 committed attempt of an
irreversible op (issue_refund, cancel_order, restock_item) on the same entity.

Run:  .venv/bin/python scripts/reference_verifier.py
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from actionproof.bench.generator import make_split  # noqa: E402  (task spec only)

APPROVAL = 50_000      # tools_ecom.APPROVAL_THRESHOLD_CENTS (restated, spec §policy)
WINDOW = 30            # tools_ecom.RETURN_WINDOW_DAYS
IRREVERSIBLE = {"issue_refund", "cancel_order", "restock_item"}
WRITE_OPS = {"issue_refund", "cancel_order", "reconcile_order", "restock_item", "update_ticket",
             "add_audit_note", "close_ticket", "escalate_to_human"}
STATUSES = {"completed", "failed", "escalated"}


def eligibility(order: dict) -> str:
    if order["status"] == "cancelled":
        return "eligible"
    if order["status"] != "returned" or order["delivered_days_ago"] > WINDOW:
        return "ineligible"
    if order["amount_cents"] > APPROVAL:
        return "requires_approval"
    return "eligible"


def task_index() -> dict:
    idx = {}
    for split in ("test_iid", "test_ood_fault", "test_iid_c", "test_ood_fault_c"):
        for t in make_split(split):
            idx[t.task_id] = t
    return idx


def verify(rec: dict, task) -> dict:
    goal = task.goal.to_json()
    st = copy.deepcopy(task.initial_state)
    orders = {o["order_id"]: o for o in st["orders"]}
    tickets = {t["ticket_id"]: t for t in st["tickets"]}
    inv = {i["sku"]: i for i in st["inventory"]}
    new_refunds: list[dict] = []          # refunds committed during the episode
    notes: Counter = Counter()
    escal: Counter = Counter()
    att = {a["action_idx"]: a for a in rec["attempts"]}
    committed: list[tuple[str, str]] = []
    policy_bad = False
    report = None

    for step in rec["trace"]:
        tool, args, i = step["tool"], step.get("args") or {}, step["idx"]
        if tool == "submit_report" and report is None and (step.get("obs") or {}).get("status") == "recorded":
            report = args.get("status")
            continue
        if tool not in WRITE_OPS or i not in att:
            continue
        a = att[i]
        assert a["op"] == tool, (rec["task_id"], i)
        if not a["committed"]:
            continue
        partial = a["fault"] == "F3_partial"
        committed.append((tool, a["entity"]))
        if tool == "issue_refund":
            o = orders[args["order_id"]]
            amt = int(args["amount_cents"])
            if not (eligibility(o) == "eligible" and 0 < amt <= o["amount_cents"]
                    and args["method"] == o["payment_method"]):
                policy_bad = True
            new_refunds.append({"order_id": o["order_id"], "amount_cents": amt, "method": args["method"]})
            if not partial:
                o["status"] = "refunded"
        elif tool == "cancel_order":
            o = orders[args["order_id"]]
            o["status"] = "cancelled"
            if not partial and o["sku"] in inv:
                inv[o["sku"]]["on_hand"] += o["qty"]
        elif tool == "reconcile_order":
            oid = args["order_id"]
            has = any(r["order_id"] == oid for r in new_refunds) or any(r["order_id"] == oid for r in st["refunds"])
            if has and oid in orders:
                orders[oid]["status"] = "refunded"
        elif tool == "restock_item":
            if args["sku"] in inv:
                inv[args["sku"]]["on_hand"] += int(args["qty"])
        elif tool == "update_ticket":
            if args["ticket_id"] in tickets:
                tickets[args["ticket_id"]]["status"] = args["status"]
        elif tool == "close_ticket":
            if args["ticket_id"] in tickets:
                tickets[args["ticket_id"]]["status"] = "closed"
        elif tool == "add_audit_note":
            notes[args["ticket_id"]] += 1
        elif tool == "escalate_to_human":
            escal[args["ticket_id"]] += 1

    # postconditions
    def holds(p: dict) -> bool:
        k = p["p"]
        if k == "refund":
            return any(r["order_id"] == p["order_id"] and r["amount_cents"] == p["amount_cents"]
                       and r["method"] == p["method"] for r in new_refunds)
        if k == "no_refund":
            return not any(r["order_id"] == p["order_id"] for r in new_refunds)
        if k == "order_status":
            return orders[p["order_id"]]["status"] == p["status"]
        if k == "audit_note":
            return notes[p["ticket_id"]] > 0
        if k == "ticket_status":
            return tickets[p["ticket_id"]]["status"] == p["status"]
        if k == "inventory":
            return inv[p["sku"]]["on_hand"] == p["on_hand"]
        if k == "escalated":
            return escal[p["ticket_id"]] > 0
        raise ValueError(f"unknown postcondition {k}")

    post = [holds(p) for p in goal["postconditions"]]
    totals: Counter = Counter()
    for r in new_refunds:
        totals[r["order_id"]] += r["amount_cents"]
    over_refund = any(totals[o] > orders[o]["amount_cents"] for o in totals)
    allowed = set(goal["allowed_entities"])
    scope_bad = any(e not in allowed for _, e in committed)
    cnt = Counter((op, e) for op, e in committed if op in IRREVERSIBLE)
    dup = any(c > 1 for c in cnt.values())
    cset = set(committed)
    coverage = all(tuple(w) in cset for w in goal["required_writes"])
    inv_bad = policy_bad or over_refund or scope_bad
    G = int(all(post) and not inv_bad and not dup and coverage)
    exp = "completed" if goal["feasible"] else "escalated"
    rep = report if report in STATUSES else None
    return {"G": G, "rep": rep, "exp": exp, "vsc": int(G == 1 and rep == exp),
            "post_ok": all(post), "policy_bad": policy_bad, "over_refund": over_refund,
            "scope_bad": scope_bad, "dup": dup, "coverage": coverage, "feasible": goal["feasible"],
            "failed_post": [p["p"] for p, ok in zip(goal["postconditions"], post) if not ok]}


def category(mine: dict, stored_vsc: int) -> str:
    why = []
    if not mine["post_ok"]:
        why.append("postcondition:" + "+".join(mine["failed_post"]))
    for k in ("policy_bad", "over_refund", "scope_bad", "dup"):
        if mine[k]:
            why.append(k)
    if not mine["coverage"]:
        why.append("no_receipt_for_required_write")
    if mine["G"] and mine["rep"] != mine["exp"]:
        why.append(f"report {mine['rep']}!={mine['exp']}")
    side = "ref=1,stored=0" if mine["vsc"] else "ref=0,stored=1"
    return f"{side}: " + (", ".join(why) if why else "ref finds nothing wrong")


def main() -> None:
    tasks = task_index()
    out = [ "# Reference verifier vs project verifier", "" ]
    groups = {"confirmatory (results/eval/conf/*_c.jsonl)": sorted(glob.glob(f"{ROOT}/results/eval/conf/*_c.jsonl")),
              "development r3 (results/eval/r3/*__test_iid.jsonl)": sorted(glob.glob(f"{ROOT}/results/eval/r3/*__test_iid.jsonl"))}
    headline = {}
    for gname, files in groups.items():
        tot = agree = 0
        cats: dict[str, list[str]] = defaultdict(list)
        rows = []
        for f in files:
            n = a = rv = sv = 0
            for line in open(f):
                rec = json.loads(line)
                mine = verify(rec, tasks[rec["task_id"]])
                stored = int(rec["vsc"])   # read ONLY for the comparison, after computing
                n += 1; rv += mine["vsc"]; sv += stored
                if mine["vsc"] == stored:
                    a += 1
                else:
                    cats[category(mine, stored)].append(f"{os.path.basename(f)}:{rec['task_id']}")
            tot += n; agree += a
            rows.append((os.path.basename(f), n, a, rv / n, sv / n))
            headline[os.path.basename(f)] = (rv / n, sv / n)
        out += [f"## {gname}", "",
                f"Agreement: **{agree}/{tot} = {agree / max(tot, 1):.4f}**", "",
                "| file | n | agree | ref VSC | stored VSC |", "|---|---|---|---|---|"]
        out += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} |" for r in rows]
        out += ["", "Disagreement categories:", ""]
        if not cats:
            out.append("None.")
        else:
            out += ["| category | count | examples |", "|---|---|---|"]
            for c, ex in sorted(cats.items(), key=lambda kv: -len(kv[1])):
                out.append(f"| {c} | {len(ex)} | {'; '.join(ex[:3])} |")
        out.append("")
        print(f"{gname}: agreement {agree}/{tot}; categories {dict((c, len(v)) for c, v in cats.items())}")

    def arm(prefix: str) -> float:
        v = [headline[k][0] for k in headline if k.startswith(prefix) and k.endswith("_c.jsonl")]
        return sum(v) / len(v)
    rft = arm("rft_tiny_v2__")
    pd0, pd1 = arm("grpo_tiny_v2_pcstr_d_s0__"), arm("grpo_tiny_v2_pcstr_d_s1__")
    out += ["## Headline check", "",
            f"Reference VSC on confirmatory (mean of test_iid_c and test_ood_fault_c, 400 tasks): "
            f"RFT v2 init **{rft:.4f}**; PCSTR-d seed 0 **{pd0:.4f}**, seed 1 **{pd1:.4f}** "
            f"(stored headline: 0.6425 -> 0.8425, both seeds).", ""]
    out += ["## Method", "",
            "The reference verifier (`scripts/reference_verifier.py`) was written from the written spec only "
            "(ARTICLE_DRAFT §2, PREREGISTRATION.md and PREREGISTRATION_A4.md §1.1, tools_ecom.py, faults.py, "
            "generator.py). It reads, per episode, `task_id`, the `trace` (tool name, arguments, observation) and "
            "the `attempts` list (action_idx, op, entity, fault, committed), and regenerates the task with "
            "`make_split` to get the initial database, postconditions, required writes, allowed entities and "
            "feasibility. It does not read or import actionproof/env/verifier*.py, actionproof/rewards or "
            "eval/a4_metrics.py, and does not read the stored `verdict`, `violations` or `vsc` when computing "
            "(`vsc` is read afterwards for the comparison only). It replays the trace on a dict copy of the "
            "initial database, applying a write only when its attempt is committed (F3 partial applies only the "
            "first sub-effect, per tools_ecom._write), checks the postconditions on the final state, three "
            "invariants (refund policy at call time, refund total <= order amount, writes only on allowed "
            "entities), duplicate irreversible writes (>1 committed issue_refund/cancel_order/restock_item on "
            "one entity) and receipt coverage of every required write; G is their conjunction and "
            "VSC = 1[G=1 and first recorded report status == ('completed' if feasible else 'escalated')]. "
            "Known simplifications: it trusts the recorded `committed`/`fault` flags of each attempt (it does not "
            "re-derive commits from the fault schedule), so it is independent of the verifier but not of the "
            "environment's commit bookkeeping; the refund postcondition is 'some committed refund row with the "
            "goal amount and method' (exactly-once is handled by the duplicate check); the bank domain is not "
            "covered; the invariant set is our reading of 'policy, exactly-once, scope' in the spec.", ""]
    with open(f"{ROOT}/results/reference_verifier.md", "w") as fh:
        fh.write("\n".join(out))
    print(f"rft {rft:.4f} pcstr_d s0 {pd0:.4f} s1 {pd1:.4f}")


if __name__ == "__main__":
    main()
