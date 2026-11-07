"""Middleware baselines (PREREGISTRATION_A4 §5.1): agent-visible reads only, tagged, never counted as the model's."""
from __future__ import annotations

from actionproof.bench.generator import make_split
from actionproof.env.tools_ecom import VERIFICATION_TOOLS
from actionproof.eval.a4_metrics import pwv
from actionproof.eval.a4_runtime import episode_record_a4, make_env_a4


def _phantom_refund_task():
    return next(t for t in make_split("val") if t.workflow == "refund" and "F1_phantom" in t.meta["faults"]
                and any(k[0] == "issue_refund" for k in t.schedule.by_call))


def _benign():
    return next(t for t in make_split("val") if t.workflow == "refund" and t.meta["benign"])


def _args(t):
    i = t.meta["ids"]
    return {"order_id": i["order"], "amount_cents": i["amount"], "method": i["method"]}


def test_raw_appends_authoritative_read_and_tags_it():
    t = _phantom_refund_task()
    env = make_env_a4(t, "r", middleware=frozenset({"raw"}))
    obs = env.call("issue_refund", _args(t))
    assert obs["status"] == "success"  # the lie is still told...
    av = obs["automatic_verification"]
    assert av[0]["tool"] == "query_ledger" and av[0]["result"]["refunds"] == []  # ...but the read exposes it
    harness = [s for s in env.trace if s.get("by")]
    assert harness and all(s["tool"] in VERIFICATION_TOOLS for s in harness)


def test_harness_reads_are_not_model_verification():
    t = _phantom_refund_task()
    env = make_env_a4(t, "r", middleware=frozenset({"raw"}))
    env.call("issue_refund", {**_args(t), "idempotency_key": "k"})
    env.call("submit_report", {"status": "completed", "summary": "x"})
    rec = episode_record_a4(t, env)
    assert pwv(env.trace, "issue_refund") is False
    assert rec["n_verify"] == 0 and rec["n_harness"] == 1 and rec["n_model_calls"] == 2


def test_vbr_blocks_only_verified_duplicates_and_injects_key():
    t = _benign()
    env = make_env_a4(t, "v", middleware=frozenset({"vbr"}))
    env.call("issue_refund", _args(t))
    assert env.trace[-1]["args"]["idempotency_key"].startswith("mw-")
    obs = env.call("issue_refund", _args(t))  # effect already visible -> blocked
    assert "already applied" in obs["note"] and env.trace[-1]["by"] == "harness_blocked"
    assert len([r for r in env.log.receipts if r.op == "issue_refund"]) == 1

    t2 = _phantom_refund_task()  # phantom: effect NOT visible -> retry is forwarded and lands exactly once
    env2 = make_env_a4(t2, "v2", middleware=frozenset({"vbr"}))
    env2.call("issue_refund", _args(t2))
    env2.call("issue_refund", _args(t2))
    assert len([r for r in env2.log.receipts if r.op == "issue_refund"]) == 1


def test_no_middleware_is_identical_to_plain_env():
    t = _phantom_refund_task()
    a, b = make_env_a4(t, "x"), make_env_a4(t, "x", middleware=frozenset())
    for env in (a, b):
        env.call("issue_refund", _args(t))
        env.call("query_ledger", {"order_id": t.meta["ids"]["order"]})
    assert [s["obs"] for s in a.trace] == [s["obs"] for s in b.trace] and type(a) is type(b)
