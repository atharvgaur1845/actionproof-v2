"""CPU tests for candidate D3 (uninformative-read local credit)."""
from __future__ import annotations

from actionproof.ad.rollout_turns import uninformative_turns
from actionproof.ad.train_irc import local_advantages


def test_uninformative_turns_requires_all_calls_uninformative_reads():
    turns = [(0, 5, [0]), (10, 15, [1]), (20, 25, [2, 3]), (30, 35, [])]
    flags = [(1, "get_journal_entries", False), (2, "list_payments", False), (3, "list_payments", True)]
    # turn 0 is a write (not in flags), turn 1 an uninformative read, turn 2 mixed, turn 3 made no call
    assert uninformative_turns(turns, flags) == [(10, 15)]


def test_local_advantages_only_touch_spans_and_never_raise_reads():
    rows = local_advantages([0.5, -0.2], [[(2, 4)], []], T=6, div=0.1, cost=0.05)
    assert rows[1] == [-0.2] * 6
    assert rows[0][:2] == [0.5, 0.5] and rows[0][4:] == [0.5, 0.5]
    assert all(abs(x - (0.5 - 0.5)) < 1e-12 for x in rows[0][2:4])  # 0.05 / 0.1 = 0.5 subtracted
