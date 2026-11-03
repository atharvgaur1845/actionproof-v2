"""DPO pairs: same task, first divergence at an assistant turn, honest failures never rejected."""
from __future__ import annotations

from actionproof.bench.prompts import NEUTRAL, VERIFY_A
from actionproof.train.dpo_data import failure_kind, first_divergence, make_pairs

SYS = {"role": "system", "content": VERIFY_A + "\n\nTOOLS"}
U = {"role": "user", "content": "task"}


def turn(a, obs="ok"):
    return [{"role": "assistant", "content": a}, {"role": "user", "content": obs}]


def rec(task, vsc, acts, I=0, D=0, F=0, obs=None):
    conv = [SYS, U]
    for i, a in enumerate(acts):
        conv += turn(a, (obs or {}).get(i, "ok"))
    return {"task_id": task, "vsc": vsc, "conversation": conv,
            "verdict": {"G": vsc, "I": I, "D": D, "F_over": F}}


def test_failure_kinds_exclude_honest_failure():
    assert failure_kind(rec("t", 1, [])) is None
    assert failure_kind(rec("t", 0, [], F=1)) == "false_completion"
    assert failure_kind(rec("t", 0, [], D=1.0, F=1)) == "duplicate"
    assert failure_kind(rec("t", 0, [], I=0.3)) == "violation"
    assert failure_kind(rec("t", 0, [])) is None  # honest failure: not a negative example


def test_pair_at_first_assistant_divergence_with_neutral_prompt():
    good = rec("t1", 1, ["lookup", "refund", "verify", "report_done"])
    bad = rec("t1", 0, ["lookup", "refund", "report_done"], F=1)
    (p,) = make_pairs([good, bad], max_per_task=3, seed=0)
    assert p["chosen"][0]["content"] == "verify" and p["rejected"][0]["content"] == "report_done"
    assert p["prompt"][0]["content"].startswith(NEUTRAL)
    assert [m["content"] for m in p["prompt"] if m["role"] == "assistant"] == ["lookup", "refund"]


def test_observation_divergence_is_skipped():
    a = rec("t1", 1, ["lookup", "x"], obs={0: "id=R-1"})
    b = rec("t1", 0, ["lookup", "y"], F=1, obs={0: "id=R-2"})
    assert first_divergence(a["conversation"], b["conversation"]) is None
    assert make_pairs([a, b], 3, 0) == []
