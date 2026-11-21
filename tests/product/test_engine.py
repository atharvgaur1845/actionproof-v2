"""Product engine checks: deterministic scenarios, timeline replay reproduces the run exactly, the divergence flag marks
exactly the faulted ok-acknowledged writes, rewards match the verifier, and the arena's recovery paths behave."""
import itertools

from actionproof.stage3.env import make_env_s3
from product.engine import FAULTS, WORKFLOWS, Arena, Scenario, build_task, run_policy, timeline


def scenarios():
    for wf, fault in itertools.product(WORKFLOWS, FAULTS):
        step = {"order_prepay": "pay", "receive_pay": "receive", "pay_post": "pay"}[wf]
        yield Scenario(workflow=wf, fault=fault, fault_step=step, ratio=0.25, seed=3)


def test_deterministic():
    for sc in scenarios():
        assert build_task(sc).to_json() == build_task(sc).to_json()
        assert run_policy(sc, "trusting") == run_policy(sc, "trusting")


def test_timeline_replays_exactly():
    from actionproof.integrity.sim import agents
    for sc in scenarios():
        t = build_task(sc)
        env = make_env_s3(t, "x", seed=0)
        agents.oracle(env, t)
        rows = timeline(t, env)
        assert [r["visible"]["response"] for r in rows] == [s["obs"] for s in env.trace]


def test_divergence_flag_and_rewards():
    for sc in scenarios():
        r = run_policy(sc, "trusting")
        flagged = any(row["truth"]["diverged"] for row in r["timeline"])
        assert flagged == (sc.fault in ("phantom", "partial", "dup_delivery", "never_commit"))
        rw = r["rewards"]
        if sc.fault in ("phantom", "partial", "never_commit"):
            assert rw["R_A (acknowledgement reward)"] == 1.0 and rw["R_V (state reward)"] == 0.0   # the observability gap
        v = run_policy(sc, "verifying")["rewards"]
        if sc.fault in ("none", "phantom", "lost_ack", "precommit_timeout", "partial"):
            assert v["VSC (evaluation metric)"] == 1.0, (sc, v)


def test_arena_partial_payment_recovery():
    a = Arena(Scenario(workflow="pay_post", fault="partial", fault_step="pay", ratio=0.25, seed=3))
    seen = a.act("read")["response"]
    assert len(seen["payments"]) == 1
    a.act("compensate")
    a.act("retry")          # same key: the store replays the stored success, nothing new is paid
    a.act("read")
    out = a.act("continue")  # post
    out = a.act("continue")  # report
    assert out["done"] and out["rewards"]["VSC (evaluation metric)"] == 0.0   # same-key retry does not re-pay


def test_arena_correct_recovery_completes():
    a = Arena(Scenario(workflow="pay_post", fault="partial", fault_step="pay", ratio=0.25, seed=3))
    a.act("read")
    a.act("compensate")      # refund the partial payment
    a.act("retry_new_key")   # pay again under a new key
    assert len([p for p in a.env.state["payments"] if p["po_id"] == a.task.meta["po_id"]]) == 3
    a.act("continue")        # post
    out = a.act("continue")  # report completed
    assert out["done"] and out["rewards"]["VSC (evaluation metric)"] == 1.0


def test_arena_blind_continue_is_false_completion():
    a = Arena(Scenario(workflow="pay_post", fault="phantom", fault_step="pay", seed=3))
    a.act("continue")
    out = a.act("continue")
    assert out["rewards"]["evidence"]["false_completion"] and out["rewards"]["R_A (acknowledgement reward)"] == 1.0
