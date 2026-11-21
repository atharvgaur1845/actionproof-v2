"""Stage 3 checks: the ratio environment equals ProcEnv without ratios; replayed starts reproduce the faulted state;
the scripted oracle reaches the top `safe` reward from every start type; the rewards rank behaviours as designed.
Needs `transformers` (system prompt rendering); skipped without it."""
from collections import defaultdict

import pytest

pytest.importorskip("transformers")

from actionproof.bench.generator_proc import DATA, load_tasks  # noqa: E402
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc  # noqa: E402
from actionproof.stage3 import rewards as R, scripted  # noqa: E402
from actionproof.stage3.env import make_env_s3  # noqa: E402
from actionproof.stage3.rollout import start_conversation  # noqa: E402

DEV = load_tasks(DATA / "s3_rec_dev.jsonl")


def run(task, policy):
    env = make_env_s3(task, "t")
    start_conversation(env, task, "")
    getattr(scripted, policy)(env, task)
    return episode_record_proc(task, env)


def test_no_ratio_equals_procenv():
    for t in load_tasks(DATA / "proc_dev.jsonl")[:40]:
        a, b = make_env_proc(t, "x"), make_env_s3(t, "x")
        from actionproof.integrity.sim.agents import oracle
        oracle(a, t)
        oracle(b, t)
        assert a.state == b.state and a.trace == b.trace


def test_replayed_start_state():
    for t in DEV:
        env = make_env_s3(t, "x")
        start_conversation(env, t, "")
        f = (t.meta.get("faults") or [{}])[0]
        if f.get("kind") == "partial":
            hit = [x for x in env.attempts if x["tool"] == f["tool"]][0]
            assert hit["committed"] == "partial"
            pays = [p["amount_cents"] for p in env.state["payments"] if p["po_id"] == t.meta["po_id"]]
            if f["tool"] == "pay_supplier":
                po = [p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"]][0]
                assert pays and 0 < pays[-1] < po["amount_cents"]
        assert not env.done


def test_oracle_reaches_top_reward_everywhere():
    by = defaultdict(list)
    for t in DEV:
        e = run(t, "oracle")
        by[t.meta["fault_class"].split(":")[1]].append(R.safe_reward(None, [0], ap_episodes=[e])[0])
    assert all(v == 1.0 for vs in by.values() for v in vs), {k: sum(v) / len(v) for k, v in by.items()}


def test_reward_ordering():
    for t in DEV:
        eo, en = run(t, "oracle"), run(t, "naive")
        so, sn = (R.safe_reward(None, [0], ap_episodes=[e])[0] for e in (eo, en))
        assert so >= sn
        kind = t.meta["fault_class"].split(":")[1]
        if kind in ("complete", "done", "lost_ack"):
            assert sn == 1.0, (t.task_id, kind)      # nothing to repair: naive completion is correct
        if kind == "dup_delivery" and t.meta["start_step"] == "pay":
            assert sn == -1.0                        # an unrefunded duplicate payment is a violation
        if kind == "partial" or kind == "phantom":
            assert sn == 0.0                         # false completion
