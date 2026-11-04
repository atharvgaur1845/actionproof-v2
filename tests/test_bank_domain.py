"""Domain B (banking, test-OOD-domain): solvable, faithful fault semantics, no leakage into prompts."""
from __future__ import annotations

from collections import Counter

from actionproof.bench.generator import make_split
from actionproof.env.db import snapshot
from actionproof.env.faults import Fault, FaultSchedule
from actionproof.env.verifier_bank import verify_bank
from actionproof.env.verifier import vsc
from actionproof.rewards.sanity_agents import bank_oracle
from actionproof.rollout.agenttune_rollout import make_env

TASKS = make_split("test_ood_domain")


def run(task, agent):
    env = make_env(task, "t", seed=0)
    agent(env)
    return env, verify_bank(task.goal, snapshot(env.conn), env.log, env.final_report)


def test_split_shape_and_fault_coverage():
    assert len(TASKS) == 150 and all(t.meta["domain"] == "bank" for t in TASKS)
    seen = Counter(f for t in TASKS for f in t.meta["faults"])
    for f in ("F1_phantom", "F2_stale", "F3_partial", "F4_timeout", "F8_ambiguous", "F5", "F6", "F7"):
        assert seen[f] > 0, f
    assert 0.25 < sum(t.meta["benign"] for t in TASKS) / len(TASKS) < 0.55


def test_oracle_solves_every_task():
    bad = [t.task_id for t in TASKS if not vsc(run(t, lambda e, t=t: bank_oracle(e, t.meta["ids"]))[1])]
    assert not bad, bad


def test_lazy_liar_fooled_by_phantom_reversal():
    t = next(t for t in TASKS if t.workflow == "dispute" and t.meta["benign"])
    t.schedule = FaultSchedule(by_call={("post_reversal", 0): Fault.PHANTOM})
    ids = t.meta["ids"]

    def liar(env):
        obs = env.call("post_reversal", {"account_id": ids["account"], "txn_id": ids["txn"], "amount_cents": ids["amount"]})
        assert obs["status"] == "success"
        env.call("close_dispute", {"dispute_id": ids["dispute"], "resolution": "done"})
        env.call("submit_report", {"status": "completed", "summary": "done"})
    _, v = run(t, liar)
    assert v.G == 0 and v.F_over == 1


def test_partial_reversal_leaves_balance_and_blind_retry_duplicates():
    t = next(t for t in TASKS if t.workflow == "dispute" and t.meta["benign"])
    ids = t.meta["ids"]
    t.schedule = FaultSchedule(by_call={("post_reversal", 0): Fault.PARTIAL})
    env, v = run(t, lambda e: e.call("post_reversal", {"account_id": ids["account"], "txn_id": ids["txn"],
                                                       "amount_cents": ids["amount"]}))
    acct = env.call("query_account", {"account_id": ids["account"]})["account"]
    assert acct["balance_cents"] == ids["balance"]  # credit not applied
    t.schedule = FaultSchedule(by_call={("post_reversal", 0): Fault.TIMEOUT})
    args = {"account_id": ids["account"], "txn_id": ids["txn"], "amount_cents": ids["amount"]}
    _, v = run(t, lambda e: (e.call("post_reversal", args), e.call("post_reversal", args)))
    assert v.D > 0


def test_prompt_is_banking_and_independent_of_fault():
    from actionproof.bench.prompts import NEUTRAL, VERIFY_A
    from actionproof.env.tools_bank import BankEnv
    p = BankEnv.adapt_prompt(NEUTRAL)
    assert "KoshBank" in p and "ShopKart" not in p and "refund" not in p.lower()
    assert "query_txn_ledger" in BankEnv.adapt_prompt(VERIFY_A)
    by_wf = {}
    for t in TASKS:
        templ = t.prompt
        for k in ("dispute", "cust", "txn", "account", "doc"):
            templ = templ.replace(str(t.meta["ids"][k]), "#")
        by_wf.setdefault(t.workflow, set()).add(templ)
    assert all(len(v) == 1 for v in by_wf.values())
