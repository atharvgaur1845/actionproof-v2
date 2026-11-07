"""A5 middleware: harness uses only the agent-visible operations log; VBR prevents the blind-retry duplicate."""
from __future__ import annotations

from actionproof.bench.generator_a5 import make_a5_split
from actionproof.eval.a5_middleware import make_env_a5_mw
from actionproof.eval.a5_runtime import episode_record_a5
from actionproof.rewards.sanity_agents_a5 import retry_blind


def _t(sc):
    return next(t for t in make_a5_split("a5_eval") if t.meta["a5"]["scenario"] == sc)


def test_vbr_prevents_blind_retry_duplicate_on_lost_ack():
    t = _t("lost_ack")
    plain = make_env_a5_mw(t, "p"); retry_blind(plain, t)
    mw = make_env_a5_mw(t, "m", middleware=frozenset({"vbr"})); retry_blind(mw, t)
    assert episode_record_a5(t, plain)["a5v"]["duplicate"] == 1
    r = episode_record_a5(t, mw)
    assert r["a5v"]["duplicate"] == 0 and r["n_harness"] >= 1
    assert all(s["tool"] in ("list_operations", "issue_refund") for s in mw.trace if s.get("by"))


def test_raw_shows_log_but_is_not_model_evidence():
    t = _t("lost_ack")
    env = make_env_a5_mw(t, "r", middleware=frozenset({"raw"}))
    i = t.meta["ids"]
    obs = env.call("issue_refund", {"order_id": i["order"], "amount_cents": i["amount"], "method": i["method"]})
    assert obs["automatic_verification"][0]["result"]["operations"][0]["status"] == "committed"
    env.call("submit_report", {"status": "completed", "summary": "x", "certificate": {}})
    assert episode_record_a5(t, env)["n_verify"] == 0
