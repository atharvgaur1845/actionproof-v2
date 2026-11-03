"""Probe example construction: positions and labels from episode records (spec §14)."""
from __future__ import annotations

from actionproof.interp.extract import build_examples


def rec(attempts, trace, conv, G=0, reported="completed", faults=("F1_phantom",)):
    return {"task_id": "t1", "conversation": conv, "attempts": attempts, "trace": trace,
            "verdict": {"G": G, "reported": reported}, "faults": list(faults)}


def tr(idx, tool):
    return {"idx": idx, "tool": tool, "args": {}, "obs": {}}


CONV = [
    {"role": "system", "content": "S"}, {"role": "user", "content": "task"},
    {"role": "assistant", "content": "lookup"}, {"role": "user", "content": "Tool result for lookup_order: {}"},
    {"role": "assistant", "content": "refund"}, {"role": "user", "content": "Tool result for issue_refund: {}"},
    {"role": "assistant", "content": "timeout-refund"}, {"role": "user", "content": "Tool result for issue_refund: {}"},
    {"role": "assistant", "content": "report"}, {"role": "user", "content": "Tool result for submit_report: {}"},
]
TRACE = [tr(0, "lookup_order"), tr(1, "issue_refund"), tr(2, "issue_refund"), tr(3, "submit_report")]


def test_phantom_ack_is_positive_pos_a_and_report_is_pos_b():
    attempts = [{"action_idx": 1, "op": "issue_refund", "entity": "O", "fault": "F1_phantom", "committed": False, "obs_status": "success"},
                {"action_idx": 2, "op": "issue_refund", "entity": "O", "fault": "F4_timeout", "committed": True, "obs_status": "error"}]
    ex = build_examples(rec(attempts, TRACE, CONV))
    a = [e for e in ex if e["pos"] == "A"]
    b = [e for e in ex if e["pos"] == "B"]
    assert len(a) == 1 and a[0]["label"] == 1 and a[0]["fault"] == "F1_phantom" and a[0]["prefix"] == 6
    assert len(b) == 1 and b[0]["label"] == 1 and b[0]["prefix"] == 8  # timeout (error obs) excluded from pos-A


def test_genuine_ack_is_negative_and_success_report_label_zero():
    attempts = [{"action_idx": 1, "op": "issue_refund", "entity": "O", "fault": None, "committed": True, "obs_status": "success"}]
    ex = build_examples(rec(attempts, TRACE, CONV, G=1, faults=()))
    assert [e["label"] for e in ex if e["pos"] == "A"] == [0]
    assert [e["label"] for e in ex if e["pos"] == "B"] == [0]
