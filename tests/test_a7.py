"""Amendment 7 gates (PREREGISTRATION_A7): reward identity, credit-demo disjointness and correctness, battery prefixes."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from actionproof.bench.generator_a4 import make_a4_split
from actionproof.bench.generator_a6 import make_demo_pool, make_depth3, make_twin_rp
from actionproof.bench.generator_a7 import make_credit_demo_pool
from actionproof.env.verifier import Verdict
from actionproof.eval.a4_runtime import episode_record_a4, make_env_a4
from actionproof.eval.forced_prefix import TAIL, _landed, _result, to_plain
from actionproof.rewards.a6_rewards import pcstr_d_noND_score, pcstr_d_noreport_score
from actionproof.rewards.a7_rewards import pcstr_d_noFunder_score
from actionproof.rewards.pcstr import pcstr_score

BASE = Verdict(G=0, P=0.0, V=0.0, I=0.0, D=0.0, F_over=1, F_under=0, reported="completed", expected="completed",
               feasible=True)


def test_noFunder_completes_the_2x2():
    for v in (BASE, replace(BASE, reported="failed", F_over=0),
              replace(BASE, G=1, P=1.0, V=1.0, F_over=0, F_under=1, reported="failed"),
              replace(BASE, G=1, P=1.0, V=1.0, F_over=0), replace(BASE, reported=None, F_over=0)):
        full, nd, fu, none = pcstr_score(v, disclosure=True), pcstr_d_noFunder_score(v), pcstr_d_noND_score(v), \
            pcstr_d_noreport_score(v)
        assert abs((full - none) - ((nd - none) + (fu - none))) < 1e-9  # additive in the two report terms
    assert pcstr_d_noFunder_score(BASE) == -0.3 and pcstr_d_noFunder_score(replace(BASE, reported="failed")) == 0
    underclaim = replace(BASE, G=1, P=1.0, V=1.0, F_over=0, F_under=1, reported="failed")
    assert pcstr_d_noFunder_score(underclaim) == pcstr_d_noreport_score(underclaim)  # F_under removed


def test_credit_demo_pool_is_disjoint_and_fault_free():
    demo = make_credit_demo_pool()
    assert len(demo) == 20 and all(t.meta["domain"] == "a4" and t.meta["a4"]["workflow"] == "credit" for t in demo)
    assert all(not t.schedule.by_call for t in demo)
    worlds = {json.dumps(t.initial_state, sort_keys=True) for t in demo}
    others = [make_a4_split(s) for s in ("a4_train_A", "a4_train_B", "e_seen_refund", "e_seen_other", "e_heldout",
                                          "e_heldout_ctrl", "e_unrecoverable")] + [make_demo_pool(), make_twin_rp(),
                                                                                    make_depth3()]
    for other in others:
        assert not worlds & {json.dumps(t.initial_state, sort_keys=True) for t in other}
        assert not {t.prompt for t in demo} & {t.prompt for t in other}


def test_credit_verifier_commits_credit_with_its_own_key():
    from actionproof.train.a7_demos import correct, credit_verifier
    for t in make_credit_demo_pool():
        env = make_env_a4(t, f"p-{t.task_id}")
        credit_verifier(env, t)
        assert len(env.trace) <= 18
        rec = episode_record_a4(t, env)
        assert correct(rec), t.task_id
        keys = {s["tool"]: s["args"].get("idempotency_key") for s in env.trace if "idempotency_key" in s["args"]}
        assert keys["apply_store_credit"] != keys["issue_refund"]  # never the refund's key (D-ENV-1)
        assert all(s["tool"] != "get_ticket" for s in env.trace)  # same template as the A6 demonstrator


def test_credit_demo_replay_through_run_episode():
    pytest.importorskip("transformers")
    from actionproof.train.a7_demos import demo
    for t in make_credit_demo_pool()[:4]:
        assert demo(t)["correct"]


def _msg(name, obs):
    return {"role": "user", "content": f"Tool result for {name}: {json.dumps(obs)}{TAIL}"}


def test_battery_schema_conversion_and_landed_edit():
    conv = [{"role": "system", "content": "A4 SYSTEM"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": '[{"tool_name": "issue_refund", "parameters": {}}]'},
            _msg("issue_refund", {"status": "success", "refund_id": "R-1", "amount_cents": 5, "method": "upi"}),
            {"role": "assistant", "content": '[{"tool_name": "query_ledger", "parameters": {}}]'},
            _msg("query_ledger", {"status": "ok", "refunds": [], "credits": []})]
    plain = to_plain(conv, "PLAIN SYSTEM")
    assert plain[0]["content"] == "PLAIN SYSTEM" and "credits" not in _result(plain[-1])[1]
    assert conv[0]["content"] == "A4 SYSTEM" and "credits" in _result(conv[-1])[1]  # source untouched
    landed = _landed(conv)
    assert _result(landed[-1])[1]["refunds"] == [{"refund_id": "R-1", "amount_cents": 5, "method": "upi",
                                                   "status": "completed"}]
    assert _result(conv[-1])[1]["refunds"] == []
