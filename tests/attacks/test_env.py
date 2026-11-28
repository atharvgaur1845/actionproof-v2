"""The attack layer: transparent without components, and each mechanic does what SPEC.md §3 says."""
from __future__ import annotations

import copy

from actionproof.attacks.backend import make_env_attack
from actionproof.attacks.generate import make_split
from actionproof.attacks.policies import policy_view
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL
from actionproof.txn.env import make_env_txn
from actionproof.txn.policies import POLICIES, Agent

DEV = make_split("attack_dev", per_family=9)


def _pick(family: str, variant: str = "attack", channel: str | None = None):
    for t in DEV:
        a = t.meta["attack"]
        if a["family"] == family and a["variant"] == variant and (channel is None or a["components"][0]["channel"]
                                                                  == channel):
            return t
    raise LookupError((family, variant, channel))


def _strip(trace):
    return [{k: v for k, v in e.items()} for e in trace]


def test_twins_and_controls_are_plain_txn_episodes():
    """Without attack components the layer changes nothing: same trace, state and attempts as TxnEnv."""
    n = 0
    for t in DEV:
        if t.meta["attack"]["variant"] == "attack" or t.meta["attack"]["schema"] == "payee_arg":
            continue
        for name in ("B0", "B4"):
            e1 = make_env_txn(t, "x", surface=False)
            e2 = make_env_attack(t, "x")
            POLICIES[name](Agent(e1, policy_view(t)))
            POLICIES[name](Agent(e2, policy_view(t)))
            assert _strip(e1.trace) == _strip(e2.trace), t.task_id
            assert e1.state == e2.state and e1.attempts == e2.attempts and e1.reports == e2.reports
            n += 1
    assert n >= 100


def test_prompt_world_and_seed_shared_within_a_pair():
    pairs: dict[str, set] = {}
    for t in DEV:
        a = t.meta["attack"]
        key = (t.prompt, t.workflow, t.meta["env_seed"], str(sorted(t.initial_state["pos"], key=lambda p: p["po_id"])))
        pairs.setdefault(a["pair_id"], set()).add(key)
    assert all(len(v) == 1 for v in pairs.values())


def test_payee_schema_only_for_f1_and_its_twin():
    t = _pick("F1_payee", "twin")
    env = make_env_attack(t, "x")
    sig = {f.__name__: f.__signature__ for f in env.tools()}
    assert "payee_account" in sig["pay_supplier"].parameters
    env2 = make_env_attack(_pick("F2_cross_po", "twin"), "x")
    assert "payee_account" not in {f.__name__: f.__signature__ for f in env2.tools()}["pay_supplier"].parameters
    po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
    obs = env.call("pay_supplier", {"po_id": po["po_id"], "amount_cents": po["amount_cents"],
                                    "payee_account": "ACCT-000000"})
    assert obs["status"] == "ok"
    assert env.state["payments"][-1]["payee_account"] == "ACCT-000000"


def test_po_note_text_reaches_the_po_read_only_in_the_attack_variant():
    t = _pick("F1_payee", "attack", "po_note")
    env = make_env_attack(t, "x")
    obs = env.call("get_purchase_order", {"po_id": t.meta["po_id"]})
    acct = t.meta["attack"]["components"][0]["params"]["acct"]
    assert acct in obs["po"]["memo"]
    assert env.trace[-1]["obs"] == obs and env.trace[-1]["attack_text"] == [0]
    twin = next(x for x in DEV if x.meta["attack"]["pair_id"] == t.meta["attack"]["pair_id"]
                and x.meta["attack"]["variant"] == "twin")
    obs2 = make_env_attack(twin, "x").call("get_purchase_order", {"po_id": t.meta["po_id"]})
    assert "memo" not in obs2["po"]


def test_f7_rollback_primary_vs_replica():
    t = _pick("F7_stale")
    step = t.meta["attack"]["components"][0]["target_step"]
    po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
    env = make_env_attack(t, "x")
    from actionproof.integrity.sim.agents import _args, _observed
    for s in t.meta["requested"]:
        if s == step:
            break
        env.call(STEP_TOOL[s], _args(s, po, f"k-{s}"))
    assert env.call(STEP_TOOL[step], _args(step, po, "k"))["status"] == "ok"
    stale = env.call(STEP_READ[step], {"po_id": po["po_id"]})
    fresh = env.call(STEP_READ[step], {"po_id": po["po_id"], "consistent": True})
    assert _observed(step, stale, po) == "held"
    assert _observed(step, fresh, po) == "absent"
    assert env.trace[-2]["txn"]["stale"] is True


def test_error_ack_is_not_executed():
    t = copy.deepcopy(_pick("F3_amount"))  # a dev task with its channel switched (no held-out task is touched)
    t.meta["attack"]["components"][0]["channel"] = "error_ack"
    po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
    env = make_env_attack(t, "x")
    n0 = len(env.state["payments"])
    obs = env.call("pay_supplier", {"po_id": po["po_id"], "amount_cents": po["amount_cents"]})
    assert obs["status"] == "error" and obs["error"] == "amount_rejected"
    assert len(env.state["payments"]) == n0 and env.trace[-1].get("forged")
    obs = env.call("pay_supplier", {"po_id": po["po_id"], "amount_cents": po["amount_cents"]})
    assert obs["status"] == "ok" and len(env.state["payments"]) == n0 + 1


def test_f6_audit_forged_against_the_record():
    for direction in ("denies", "confirms"):
        t = next(x for x in DEV if x.meta["attack"]["family"] == "F6_evidence" and x.meta["attack"]["variant"] ==
                 "attack" and x.meta["attack"]["components"][0]["params"]["direction"] == direction)
        c = t.meta["attack"]["components"][0]
        po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
        env = make_env_attack(t, "x")
        from actionproof.integrity.sim.agents import _args, _observed
        for s in t.meta["requested"]:
            ack = env.call(STEP_TOOL[s], _args(s, po, f"k-{s}"))
            if s == c["target_step"]:
                assert ack["error"] == "timeout"
                break
        log = env.call("get_audit_log", {"po_id": po["po_id"]})
        e = [x for x in log["requests"] if x["tool"] == STEP_TOOL[c["target_step"]]][0]
        rec = env.call(STEP_READ[c["target_step"]], {"po_id": po["po_id"], "consistent": True})
        seen = _observed(c["target_step"], rec, po)
        if direction == "denies":
            assert e["ack_status"] == "rejected" and e["committed"] is False and seen == "held"
        else:
            assert e["ack_status"] == "ok" and e["committed"] is True and seen == "absent"


def test_gate_sees_the_attacked_reply():
    """With the gate on, the gate's own reads pass through the attack layer too (the gate is not given clean data)."""
    t = _pick("F7_stale")
    env = make_env_attack(t, "x", rules=__import__("actionproof.recovery_gate.gate", fromlist=["Rules"]).Rules())
    cls = [c.__name__ for c in type(env).__mro__]
    assert cls.index("GateMixin") < cls.index("AttackMixin") < cls.index("TxnEnv")
    assert copy.deepcopy(env.atk)[0]["family"] == "F7_stale"
