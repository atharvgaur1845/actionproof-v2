"""Multi-axis fault taxonomy (PLAN.md section G1), computed from the task specification only. CPU.

    python scripts/classify_faults.py                       # c2 splits -> data/bench/c2/fault_labels.jsonl + .md
    python scripts/classify_faults.py --tasks a.jsonl b.jsonl --out labels.jsonl

Inputs are ONLY the task spec: the fault schedule, the goal (postconditions, required writes, granted entities),
and meta (fault list, precommit_timeouts, env_version, the A4 environment class when present, F5 feasibility).
No episode, outcome, model output or verifier result is read. The output is deterministic (sorted, canonical JSON).

Definitions used by the rules
- Sub-effects of each write op (what a commit changes):
  issue_refund: refund_row, order_refunded; cancel_order: order_cancelled, inventory_restored;
  reconcile_order: order_refunded; restock_item: inventory_added; add_audit_note: audit_note;
  close_ticket / update_ticket: ticket_status; escalate_to_human: escalation; apply_store_credit: credit_row.
- Authoritative projection of each sub-effect (a fresh read of the system of record, synchronous commits):
  refund_row -> refunds table (query_ledger); order_* -> orders table (lookup_order); inventory_* -> inventory
  (check_inventory); audit_note, ticket_status -> tickets and audit notes (get_ticket); escalation -> escalations
  table (gateway projection only, no agent read tool); credit_row -> credits (query_ledger, A4 env).
  `get_refund_status` is a cached endpoint (F6) and is never counted as authoritative.
- Environment class: the A4 meta value when present; otherwise derived from the schedule: OK (no call fault),
  UR (some write tool fails on every call index below the horizon), RP (call faults that stop before the horizon).
  Horizon = 12 calls (the A11 evaluation turn budget).
- Commit state of a call fault: F1 phantom, F2 stale, F4 timeout listed in precommit_timeouts: not committed.
  F4 timeout otherwise: committed, error ack. F3 partial: first sub-effect committed only. F8 ambiguous: the
  schedule's commits flag.

Axis 1, state observability (Full / Partial / None). Over the sub-effects of every required write and every
faulted op: Full if each has an authoritative projection; None if none has one; Partial otherwise.

Axis 2, contract coverage (Complete / Incomplete). The contract is the goal spec plus the verifier's invariants
(policy validity, refund within order amount, granted entities) and the duplicate-irreversible term. A sub-effect is
covered if a postcondition predicate states it on the same entity, or it is the primary (first) sub-effect of a
required write (receipt coverage). Task faults: F5 is covered by the no_refund and escalated postconditions, F6 by
the refund postcondition (ledger, not the status endpoint), F7 by the invariants plus the duplicate term.
Complete if every sub-effect of every faulted op and every task fault is covered; Incomplete otherwise.

Axis 3, recoverability (Safe / Conditional / Impossible). Per faulted op, then the worst over ops
(Impossible > Conditional > Safe):
- Impossible if the op fails on every call within the horizon (environment class UR).
- Partial commit: the repair op for the missing sub-effect decides (issue_refund -> reconcile_order, which is
  state-idempotent: Safe; cancel_order -> restock_item, irreversible without a key: Conditional).
- Otherwise (no commit, unknown commit, or committed with an error ack) the recovery is to re-issue the op only if
  it is absent. Safe if a re-issue cannot create a second irreversible effect whatever the hidden commit state:
  the op takes an idempotency key and the environment scopes keys per (op, entity, key) (env_version 2), or the op
  sets state idempotently (close_ticket, update_ticket, reconcile_order, cancel_order which is guarded by order
  status), or the op is reversible (add_audit_note, escalate_to_human). Conditional if the op is irreversible and
  only safe under a provider assumption (restock_item; issue_refund under v1 global keys).
- A task with no call fault is Safe (no recovery is needed; F5 escalation, F6 and F7 have valid actions).

Axis 4 (agent observability) is set by the gateway mode, not by the task, and is not labelled here. The extra
field `agent_read_coverage` records whether each required sub-effect also has an agent-visible read tool.

Cells that no task falls into are reported as empty (count 0) in the summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HORIZON = 12
SUB_EFFECTS = {
    "issue_refund": ("refund_row", "order_refunded"), "cancel_order": ("order_cancelled", "inventory_restored"),
    "reconcile_order": ("order_refunded",), "restock_item": ("inventory_added",), "add_audit_note": ("audit_note",),
    "close_ticket": ("ticket_status",), "update_ticket": ("ticket_status",), "escalate_to_human": ("escalation",),
    "apply_store_credit": ("credit_row",),
}
AUTHORITATIVE = {"refund_row": "query_ledger", "order_refunded": "lookup_order", "order_cancelled": "lookup_order",
                 "inventory_restored": "check_inventory", "inventory_added": "check_inventory",
                 "audit_note": "get_ticket", "ticket_status": "get_ticket", "escalation": "escalations_table",
                 "credit_row": "query_ledger"}
AGENT_READ = {k: v for k, v in AUTHORITATIVE.items() if v != "escalations_table"}
# sub-effect -> postcondition predicate that states it
STATED_BY = {"refund_row": "refund", "order_refunded": "order_status", "inventory_restored": "inventory",
             "inventory_added": "inventory", "audit_note": "audit_note", "ticket_status": "ticket_status",
             "escalation": "escalated", "credit_row": "credit"}
KEYED = {"issue_refund", "apply_store_credit"}
STATE_IDEMPOTENT = {"close_ticket", "update_ticket", "reconcile_order", "cancel_order"}
REVERSIBLE = {"add_audit_note", "escalate_to_human", "close_ticket", "update_ticket", "reconcile_order"}
REPAIR = {"issue_refund": "reconcile_order", "cancel_order": "restock_item"}
TASK_FAULT_COVER = {"F5": ("no_refund", "escalated"), "F6": ("refund",), "F7": ()}
ORDER = {"Safe": 0, "Conditional": 1, "Impossible": 2}
AXES = {"state_observability": ("Full", "Partial", "None"), "contract_coverage": ("Complete", "Incomplete"),
        "recoverability": ("Safe", "Conditional", "Impossible")}


def env_class(d: dict) -> str:
    a4 = (d.get("meta") or {}).get("a4") or {}
    if a4.get("env_class"):
        return a4["env_class"]
    by_tool: dict[str, set[int]] = {}
    for x in d["schedule"]:
        by_tool.setdefault(x["tool"], set()).add(int(x["k"]))
    if not by_tool:
        return "OK"
    return "UR" if any(set(range(HORIZON)) <= ks for ks in by_tool.values()) else "RP"


def commit_state(d: dict, x: dict) -> str:
    pre = {tuple(p) for p in (d.get("meta") or {}).get("precommit_timeouts", [])}
    f = x["fault"]
    if f in ("F1_phantom", "F2_stale") or (f == "F4_timeout" and (x["tool"], int(x["k"])) in pre):
        return "none"
    if f == "F3_partial":
        return "partial"
    if f == "F8_ambiguous":
        return "full" if x.get("commits") else "none"
    return "full"  # F4 timeout after commit


def _entity_of(op: str, ids: dict) -> str | None:
    if op in ("issue_refund", "cancel_order", "reconcile_order", "apply_store_credit"):
        return ids.get("order")
    if op == "restock_item":
        return ids.get("sku")
    return ids.get("ticket")


def covered(sub: str, op: str, ent: str | None, goal: dict) -> bool:
    if [op, ent] in [list(w) for w in goal["required_writes"]] and SUB_EFFECTS[op][0] == sub:
        return True  # primary sub-effect of a required write: receipt coverage
    want = STATED_BY.get(sub)
    for pc in goal["postconditions"]:
        if pc["p"] == want and ent in (pc.get("order_id"), pc.get("ticket_id"), pc.get("sku"), None):
            return True
        if pc["p"] == "inventory" and sub in ("inventory_restored", "inventory_added"):
            return True  # the SKU of the target order
    return False


def op_recoverability(op: str, cls: str, state: str, keys_scoped: bool) -> str:
    if cls == "UR":
        return "Impossible"
    if state == "partial":
        rep = REPAIR.get(op)
        return "Safe" if rep in STATE_IDEMPOTENT or rep in REVERSIBLE else "Conditional"
    if op in KEYED:
        return "Safe" if keys_scoped else "Conditional"
    if op in STATE_IDEMPOTENT or op in REVERSIBLE:
        return "Safe"
    return "Conditional"


def label(d: dict) -> dict:
    meta, goal = d.get("meta") or {}, d["goal"]
    ids = meta.get("ids") or {}
    faults = sorted(meta.get("faults") or [])
    cls = env_class(d)
    keys_scoped = meta.get("env_version") == 2
    faulted = sorted({(x["tool"], commit_state(d, x)) for x in d["schedule"]})
    required = [tuple(w) for w in goal["required_writes"]]

    subs = [(op, s) for op, _ in required for s in SUB_EFFECTS.get(op, ())] + \
           [(op, s) for op, _ in faulted for s in SUB_EFFECTS.get(op, ())]
    has_proj = [s in AUTHORITATIVE for _, s in subs]
    obs = "Full" if all(has_proj) else ("None" if not any(has_proj) else "Partial")
    agent_read = "all" if all(s in AGENT_READ for _, s in subs) else \
        ("none" if not any(s in AGENT_READ for _, s in subs) else "some")

    cov = all(covered(s, op, _entity_of(op, ids), goal) for op, _ in faulted for s in SUB_EFFECTS.get(op, ()))
    preds = {pc["p"] for pc in goal["postconditions"]}
    for tf in (f for f in faults if "_" not in f):  # task faults (F5, F6, F7) carry no suffix
        need = TASK_FAULT_COVER.get(tf)
        cov &= need is not None and all(p in preds for p in need)

    recs = [op_recoverability(op, cls, st, keys_scoped) for op, st in faulted]
    recov = max(recs, key=ORDER.__getitem__) if recs else "Safe"
    return {"task_id": d["task_id"], "split": d.get("split"), "workflow": d.get("workflow"), "faults": faults,
            "faulted_ops": [[op, st] for op, st in faulted], "env_class": cls, "feasible": goal["feasible"],
            "env_version": meta.get("env_version", 1), "precommit": bool(meta.get("precommit_timeouts")),
            "state_observability": obs, "contract_coverage": "Complete" if cov else "Incomplete",
            "recoverability": recov, "agent_read_coverage": agent_read,
            "cell": f"{obs}/{'Complete' if cov else 'Incomplete'}/{recov}"}


def summary_table(labels: list[dict]) -> str:
    lines = ["| split | state observability | contract coverage | recoverability | tasks |", "|---|---|---|---|---|"]
    splits = sorted({l["split"] for l in labels})
    for sp in splits + ["all"]:
        sub = [l for l in labels if sp == "all" or l["split"] == sp]
        c = Counter((l["state_observability"], l["contract_coverage"], l["recoverability"]) for l in sub)
        for o in AXES["state_observability"]:
            for cv in AXES["contract_coverage"]:
                for r in AXES["recoverability"]:
                    n = c.get((o, cv, r), 0)
                    lines.append(f"| {sp} | {o} | {cv} | {r} | {n if n else 'empty'} |")
    lines += ["", "Recoverability by environment class and fault (all splits):", "",
              "| env class | faults | recoverability | tasks |", "|---|---|---|---|"]
    c2 = Counter((l["env_class"], "+".join(l["faults"]) or "benign", l["recoverability"]) for l in labels)
    lines += [f"| {a} | {b} | {r} | {n} |" for (a, b, r), n in sorted(c2.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["data/bench/c2/test_iid_c2.jsonl", "data/bench/c2/v2_stress_c2.jsonl"])
    ap.add_argument("--out", default="data/bench/c2/fault_labels.jsonl")
    a = ap.parse_args()
    labels = [label(json.loads(line)) for p in a.tasks for line in (ROOT / p).open()]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(l, sort_keys=True) + "\n" for l in labels))
    table = summary_table(labels)
    out.with_suffix(".md").write_text(f"# Fault taxonomy labels ({', '.join(a.tasks)})\n\n"
                                      "Generated by scripts/classify_faults.py from task specs only.\n\n" + table)
    print(table)


if __name__ == "__main__":
    main()
