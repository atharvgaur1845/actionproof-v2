"""ActionProof-Bench v2 environment semantics: scoped idempotency (fixes D-ENV-1) and timeout before commit."""
import json

from actionproof.bench.generator import Task, make_split
from actionproof.env.faults import Fault
from actionproof.rollout.agenttune_rollout import make_env


def _v2(precommit: bool):
    t = next(t for t in make_split("test_iid") if t.schedule.by_call and
             list(t.schedule.by_call.values())[0] is Fault.TIMEOUT and list(t.schedule.by_call)[0][0] == "close_ticket")
    d = t.to_json()
    d["meta"] = {**d["meta"], "env_version": 2,
                 "precommit_timeouts": [[x["tool"], x["k"]] for x in d["schedule"]] if precommit else []}
    return Task.from_json(json.loads(json.dumps(d)))


def _close(env, ticket):
    return env.call("close_ticket", {"ticket_id": ticket})


def _ticket(t):
    return next(p["ticket_id"] for p in t.goal.postconditions if p["p"] == "ticket_status")


def test_timeout_after_commit_commits():
    t = _v2(False)
    env = make_env(t, "r")
    obs = _close(env, _ticket(t))
    assert obs["status"] == "error" and env.log.attempts[-1].committed


def test_timeout_before_commit_does_not_commit_and_retry_lands():
    t = _v2(True)
    env = make_env(t, "r")
    first = _close(env, _ticket(t))
    assert first["status"] == "error" and not env.log.attempts[-1].committed
    second = _close(env, _ticket(t))
    assert second["status"] == "success" and env.log.attempts[-1].committed


def test_v1_task_uses_v1_env():
    t = make_split("test_iid")[0]
    assert type(make_env(t, "r")).__name__ == "EcomEnv"
