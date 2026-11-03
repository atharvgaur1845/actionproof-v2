"""RFT data builder: VSC filter, neutral-prompt swap, dedup, per-turn rows."""
from __future__ import annotations

import pytest

from actionproof.bench.prompts import NEUTRAL, VERIFY_A
from actionproof.train.rft_data import action_signature, neutralise, select, turn_rows

TOOLS = "\n\nYou can call these tools:\n[...]"


def rec(task, vsc, acts, sample=0):
    conv = [{"role": "system", "content": VERIFY_A + TOOLS}, {"role": "user", "content": f"do {task}"}]
    for a in acts:
        conv += [{"role": "assistant", "content": a}, {"role": "user", "content": "Tool result for x: {}"}]
    return {"task_id": task, "vsc": vsc, "sample": sample, "conversation": conv}


def test_neutralise_swaps_prefix_keeps_tools():
    s = neutralise(VERIFY_A + TOOLS)
    assert s == NEUTRAL + TOOLS and "Reliability rules" not in s
    with pytest.raises(ValueError):
        neutralise("something else")


def test_turn_rows_one_per_assistant_turn_with_history():
    rows = turn_rows(rec("t1", 1, ["a1", "a2", "a3"])["conversation"], {"task_id": "t1"})
    assert [r["completion"][0]["content"] for r in rows] == ["a1", "a2", "a3"]
    assert rows[2]["prompt"][0]["content"].startswith(NEUTRAL)
    assert [m["content"] for m in rows[2]["prompt"] if m["role"] == "assistant"] == ["a1", "a2"]


def test_select_filters_dedups_and_caps():
    recs = [rec("t1", 1, ["a"]), rec("t1", 1, ["a"], 1), rec("t1", 1, ["b"], 2), rec("t1", 1, ["c"], 3),
            rec("t2", 0, ["z"]), rec("t3", 1, ["q"])]
    cur = select(recs, raw=False, per_task_cap=2, n_target=None, seed=0)
    assert all(r["vsc"] == 1 for r in cur)
    assert len({action_signature(r["conversation"]) for r in cur}) == len(cur)
    assert sum(r["task_id"] == "t1" for r in cur) == 2
    raw = select(recs, raw=True, per_task_cap=2, n_target=len(cur), seed=0)
    assert len(raw) == len(cur)
