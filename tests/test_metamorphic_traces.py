"""Metamorphic tests on RECORDED evaluation episodes (Amendment 8 addendum triage; complements test_exploits.py,
which uses scripted agents). Each recorded episode is re-executed on a fresh simulator and then perturbed:

  (a) the report's free-text summary replaced by unrelated text  -> G, P, V, I, D unchanged;
  (b) only the report status changed                             -> G, P, V, I, D unchanged (only report terms move);
  (c) read-only calls inserted before every write and the report -> every recorded observation still matches, and the
      final database snapshot and G, P, V, I, D are unchanged (a read never changes state or later acknowledgements,
      the assumption behind ARTICLE §3).

Fixed sample: the first 40 episodes of three recorded v2 test_iid files (warm start, PCSTR-d, tool_trust).
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from actionproof.bench.generator import make_split
from actionproof.env.counterfactual import replay
from actionproof.env.db import snapshot
from actionproof.env.verifier import verify
from actionproof.rollout.agenttune_rollout import make_env

ROOT = Path(__file__).resolve().parents[1]
FILES = ["results/eval/r3/rft_tiny_v2__neutral__test_iid.jsonl",
         "results/eval/r3/grpo_tiny_v2_pcstr_d_s0__neutral__test_iid.jsonl",
         "results/eval/r3/grpo_tiny_v2_tool_trust_s0__neutral__test_iid.jsonl"]
N_PER_FILE = 40
TASKS = {t.task_id: t for t in make_split("test_iid")}
STATE = ("G", "P", "V", "I", "D")
READS = {"order_id": ("lookup_order", "query_ledger"), "ticket_id": ("get_ticket",), "sku": ("check_inventory",)}


def _episodes():
    out = []
    for f in FILES:
        p = ROOT / f
        if not p.exists():
            continue
        with open(p) as fh:
            for i, line in enumerate(fh):
                if i >= N_PER_FILE:
                    break
                r = json.loads(line)
                if r["task_id"] in TASKS and all(not s.get("by") for s in r["trace"]):
                    out.append(r)
    return out


EPISODES = _episodes()
pytestmark = pytest.mark.skipif(not EPISODES, reason="recorded test_iid files not present")


def _state(task, env):
    v = verify(task.goal, snapshot(env.conn), env.log, env.final_report)
    return tuple(getattr(v, k) for k in STATE), v


def _with_report(rec, **change):
    r = copy.deepcopy(rec)
    for s in r["trace"]:
        if s["tool"] == "submit_report" and isinstance(s.get("args"), dict):
            s["args"].update(change)
    return r


@pytest.mark.parametrize("rec", EPISODES, ids=lambda r: f"{r.get('model', '?')}:{r['task_id']}")
def test_summary_text_cannot_move_state_terms(rec):
    task = TASKS[rec["task_id"]]
    base, _ = _state(task, replay(task, rec))
    noise = " ".join(random.Random(rec["task_id"]).choice(["refund", "applied", "verified", "done", "ledger", "x"])
                     for _ in range(12))
    assert _state(task, replay(task, _with_report(rec, summary=noise)))[0] == base


@pytest.mark.parametrize("rec", EPISODES, ids=lambda r: f"{r.get('model', '?')}:{r['task_id']}")
def test_report_status_moves_only_report_terms(rec):
    task = TASKS[rec["task_id"]]
    if not any(s["tool"] == "submit_report" for s in rec["trace"]):
        pytest.skip("episode has no report")
    base, vb = _state(task, replay(task, rec))
    for status in ("completed", "failed", "escalated"):
        st, v = _state(task, replay(task, _with_report(rec, status=status)))
        assert st == base
        assert v.reported == status


@pytest.mark.parametrize("rec", EPISODES, ids=lambda r: f"{r.get('model', '?')}:{r['task_id']}")
def test_inserted_reads_change_nothing(rec):
    task = TASKS[rec["task_id"]]
    ref = replay(task, rec)
    base, _ = _state(task, ref)
    ids = task.meta["ids"]
    reads = [(tool, {arg: ids[key]}) for arg, key in (("order_id", "order"), ("ticket_id", "ticket"), ("sku", "sku"))
             if ids.get(key) for tool in READS[arg]]
    assert len(reads) >= 3
    env = make_env(task, rec.get("rollout_id", "replay"), seed=0)
    for s in rec["trace"]:
        for tool, args in reads:  # read everything before every recorded call
            assert env.call(tool, args).get("status") == "ok"
        assert env.call(s["tool"], s["args"]) == s["obs"], f"{s['tool']} observation changed after inserted reads"
    assert snapshot(env.conn) == snapshot(ref.conn)
    assert _state(task, env)[0] == base
