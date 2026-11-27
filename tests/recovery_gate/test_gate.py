"""Recovery Gate (RG1) tests: no-gate equivalence, no hidden-state access, intent visibility, counterfactual blindness,
and one golden case per rule."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.recovery_gate import gate as G
from actionproof.recovery_gate.env import make_env_gated, run_policy_gated
from actionproof.txn.env import TxnEnv
from actionproof.txn.policies import POLICIES, run_policy

ROOT = Path(__file__).resolve().parents[2]
DEV = load_tasks(DATA / "txn_dev.jsonl")
VAL = load_tasks(DATA / "txn_val.jsonl")


def find(pred):
    return next(t for t in DEV + VAL if pred(t))


def fault(t):
    return (t.meta.get("faults") or [{}])[0]


def po(t):
    return G.intent_of(t)


def env_for(t, rules=G.Rules()):
    return make_env_gated(t, "test", rules=rules, surface=True)


def pay_args(t, key="K1"):
    i = po(t)
    return {"po_id": i.po_id, "amount_cents": i.amount_cents, "idempotency_key": key}


def agent_attempts(e, tool):
    return [a for a in e.attempts if a["tool"] == tool and a.get("by", "agent") == "agent"]


# ---------------------------------------------------------------------- equivalence and leakage
@pytest.mark.parametrize("name", ["B0", "B1", "B3", "B4"])
def test_no_gate_env_is_the_txn_env(name):
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.txn import policies, truth
    for t in DEV[:60]:
        ref = run_policy(name, t)
        e = make_env_gated(t, f"txnpol-{name}-{t.task_id}", rules=None, surface=False)
        a = policies.Agent(e, t)
        try:
            POLICIES[name](a)
        except policies.BudgetExceeded:
            pass
        rec = episode_record_proc(t, e)
        assert rec["trace"] == ref["trace"] and rec["attempts"] == ref["attempts"] and rec["report"] == ref["report"]
        assert truth.txn_score(rec) == ref["txn"]


FORBIDDEN = {"state", "attempts", "fault_plan", "txn_plan", "trace", "idem", "pending", "lag", "never_commit",
             "faults", "reality", "regime", "escalation_correct", "txn", "unavailable", "version", "prev", "calls",
             "reports", "audit", "txn_cfg", "_read_meta", "final_state"}


def test_gate_module_reads_no_hidden_state():
    src = (ROOT / "actionproof/recovery_gate/gate.py").read_text()
    tree = ast.parse(src)
    allowed_fn = {"intent_of"}  # reads the PO and the requested steps, which the prompt states (next test)
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]:
        if fn.name in allowed_fn:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute):
                assert node.attr not in FORBIDDEN, f"{fn.name} accesses .{node.attr}"
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                assert node.slice.value not in {"faults", "reality", "regime", "escalation_correct", "txn",
                                                "unavailable_reads"}, f"{fn.name} reads [{node.slice.value!r}]"
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods <= {"__future__", "json", "dataclasses", "typing", "actionproof.integrity.sim.agents",
                    "actionproof.integrity.sim.env"}, mods
    assert set(inspect.signature(G.Gate.__init__).parameters) == {"self", "intent", "rules", "name"}


def test_gate_holds_no_env_reference():
    t = find(lambda t: fault(t).get("kind") == "lost_ack" and fault(t).get("tool") == "pay_supplier")
    e = env_for(t)
    e.call("pay_supplier", pay_args(t))
    e.call("pay_supplier", pay_args(t))
    e.call("submit_report", {"status": "completed", "summary": "x"})

    def walk(x, depth=0):
        assert not isinstance(x, TxnEnv)
        assert not (inspect.ismethod(x) and isinstance(getattr(x, "__self__", None), TxnEnv))
        if depth > 6:
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v, depth + 1)
        elif hasattr(x, "__dict__") and not inspect.isfunction(x):
            for v in vars(x).values():
                walk(v, depth + 1)
    walk(e.gate)


@pytest.mark.parametrize("split", ["txn_dev", "txn_val", "txn_test", "rg1_test"])
def test_intent_is_stated_in_the_prompt(split):
    for t in load_tasks(DATA / f"{split}.jsonl"):
        i = G.intent_of(t)
        assert i.po_id in t.prompt and f"{i.qty} x" in t.prompt and str(i.amount_cents) in t.prompt, t.task_id


def test_counterfactual_blindness_on_quads():
    """Within a quadruple the prompt is identical. The gate's calls may differ only after the observations differ."""
    quads: dict[str, list] = {}
    for t in DEV + VAL:
        q = t.meta["txn"].get("quad_id")
        if q:
            quads.setdefault(q, []).append(t)
    assert quads
    for name in ("B0", "B3"):
        for q, ts in quads.items():
            recs = [run_policy_gated(name, t) for t in ts]
            for r1 in recs:
                for r2 in recs:
                    a = [(s["tool"], s["args"]) for s in r1["trace"] if s.get("by", "agent") == "agent"]
                    b = [(s["tool"], s["args"]) for s in r2["trace"] if s.get("by", "agent") == "agent"]
                    oa = [s["obs"] for s in r1["trace"] if s.get("by", "agent") == "agent"]
                    ob = [s["obs"] for s in r2["trace"] if s.get("by", "agent") == "agent"]
                    k = next((i for i, (x, y) in enumerate(zip(oa, ob)) if x != y), min(len(oa), len(ob)))
                    assert a[:k + 1] == b[:k + 1], (q, name)


# ---------------------------------------------------------------------- one golden case per rule
def test_verify_rejects_a_completed_report_on_a_phantom_write():
    t = find(lambda t: fault(t).get("kind") == "phantom" and fault(t).get("tool") == "pay_supplier"
             and t.meta["txn"]["regime"] == "standard" and not t.meta["txn"].get("surface"))
    e = env_for(t)
    assert e.call("pay_supplier", pay_args(t))["status"] == "ok"  # the phantom ack
    r = e.call("submit_report", {"status": "completed", "summary": "paid"})
    assert r["error"] == "report_rejected_by_gate" and not e.reports
    assert any(s.get("gate") == "gate_verify" and s["tool"] == "list_payments" for s in e.trace)
    assert e.call("pay_supplier", pay_args(t))["status"] == "ok"  # record read since: absent, so the resend is sent


def test_read_before_retry_and_no_resend_of_a_held_step():
    t = find(lambda t: fault(t).get("kind") == "lost_ack" and fault(t).get("tool") == "pay_supplier"
             and not t.meta["txn"].get("surface"))
    e = env_for(t)
    assert e.call("pay_supplier", pay_args(t))["error"] == "timeout"
    r = e.call("pay_supplier", pay_args(t, "K2"))
    assert r["error"] == "not_sent_by_gate" and len(agent_attempts(e, "pay_supplier")) == 1
    assert r["record"][0]["tool"] == "list_payments"
    r = e.call("pay_supplier", pay_args(t, "K3"))
    assert r["error"] == "blocked_by_gate" and r["rule"] == "retry" and len(agent_attempts(e, "pay_supplier")) == 1


def test_no_retry_ablation_sends_the_duplicate():
    t = find(lambda t: fault(t).get("kind") == "lost_ack" and fault(t).get("tool") == "pay_supplier"
             and not t.meta["txn"].get("surface"))
    e = env_for(t, G.ABLATIONS["no_retry"])
    e.call("pay_supplier", pay_args(t))
    e.call("pay_supplier", pay_args(t, "K2"))
    assert len(agent_attempts(e, "pay_supplier")) == 2


def test_escalate_when_the_record_cannot_be_read():
    t = find(lambda t: t.meta["txn"]["regime"] == "unresolvable" and t.meta["txn"]["quad_step"] == "pay")
    e = env_for(t)
    e.call("pay_supplier", pay_args(t))
    assert e.call("list_payments", {"po_id": po(t).po_id})["error"] == "service_unavailable"
    r = e.call("pay_supplier", pay_args(t))
    assert r["rule"] == "escalate" and len(agent_attempts(e, "pay_supplier")) == 1
    e.call("submit_report", {"status": "completed", "summary": "done"})
    assert e.reports[0]["status"] == "escalated"
    assert any(s["tool"] == "escalate_to_human" and s.get("gate") == "gate_escalate" for s in e.trace)


def test_partial_receipt_is_topped_up_by_the_missing_quantity_only():
    t = find(lambda t: fault(t).get("kind") == "partial" and fault(t).get("tool") == "record_goods_receipt"
             and not t.meta["txn"].get("surface"))
    i = po(t)
    e = env_for(t)
    e.call("record_goods_receipt", {"po_id": i.po_id, "qty": i.qty})
    obs = e.call("list_goods_receipts", {"po_id": i.po_id})
    net = sum(r["qty"] for r in obs["receipts"] if r["kind"] == "receipt")
    assert 0 < net < i.qty
    r = e.call("record_goods_receipt", {"po_id": i.po_id, "qty": i.qty})
    assert r["rule"] == "repair" and str(i.qty - net) in r["detail"]
    assert e.call("record_goods_receipt", {"po_id": i.po_id, "qty": i.qty - net})["status"] == "ok"


def test_the_only_correct_payment_is_never_refunded():
    t = find(lambda t: not t.meta.get("faults") or fault(t).get("tool") != "pay_supplier")
    t = find(lambda t: "pay" in t.meta["requested"] and all(f["tool"] != "pay_supplier" for f in t.meta.get("faults") or [])
             and t.meta["txn"]["regime"] == "standard" and not t.meta["txn"].get("surface"))
    i = po(t)
    e = env_for(t)
    pid = e.call("pay_supplier", pay_args(t))["payment_id"]
    r = e.call("refund_payment", {"payment_id": pid})
    assert r["rule"] == "repair" and not agent_attempts(e, "refund_payment")
    assert i.po_id


def test_intent_binding_blocks_a_wrong_amount():
    t = find(lambda t: "pay" in t.meta["requested"] and not t.meta["txn"].get("surface"))
    e = env_for(t)
    r = e.call("pay_supplier", {**pay_args(t), "amount_cents": po(t).amount_cents + 1})
    assert r["rule"] == "intent" and not agent_attempts(e, "pay_supplier")


def test_finalize_completes_only_with_a_confirming_record():
    t = find(lambda t: t.meta["requested"] == ["pay", "post"] if False else
             (all(f["tool"] != "pay_supplier" for f in t.meta.get("faults") or []) and t.meta["requested"] == ["pay"]))\
        if any(x.meta["requested"] == ["pay"] for x in DEV + VAL) else None
    if t is None:  # no single-step pay task in dev/val: use any fault-free pay step of a multi-step task
        t = find(lambda t: not t.meta.get("faults") and t.meta["txn"]["regime"] == "standard"
                 and not t.meta["txn"].get("surface"))
    e = env_for(t)
    e.gate_finalize()
    assert e.reports and e.reports[0]["status"] == "escalated"  # nothing was done: no completion claim


def test_alias_surface_messages_use_the_surface_names():
    import copy
    t = copy.deepcopy(find(lambda t: fault(t).get("kind") == "lost_ack" and fault(t).get("tool") == "pay_supplier"))
    t.meta["txn"]["surface"] = "alias"  # dev/val have no OOD groups; put a dev task on the alias surface
    e = env_for(t)
    pay = e.c2t["pay_supplier"]
    args = pay_args(t)
    assert e.call(pay, args)["error"] == "timeout"
    r = e.call(pay, {**args, "idempotency_key": "K2"})
    assert r["error"] == "not_sent_by_gate"
    assert e.c2t["list_payments"] in r["detail"] and "list_payments" not in r["detail"]
    assert r["record"][0]["tool"] == e.c2t["list_payments"] and len(agent_attempts(e, "pay_supplier")) == 1
