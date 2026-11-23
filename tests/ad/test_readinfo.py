"""CPU tests for per-read information flags."""
from __future__ import annotations

from actionproof.ad.readinfo import read_flags
from actionproof.ad.scripted import POLICIES
from actionproof.ad.worlds import make_env_cw
from actionproof.bench.generator_proc import DATA, load_tasks

TASKS = load_tasks(DATA / "cw_dev.jsonl")[:30]


def flags(task, pol="always"):
    env = make_env_cw(task, "t", seed=7)
    POLICIES[pol](env, task)
    return read_flags(task, env.trace, 7), env.trace


def test_journal_read_is_uninformative_and_first_write_read_is_informative():
    for t in TASKS:
        fl, trace = flags(t)
        tools = [tool for _, tool, _ in fl]
        assert tools, "the always policy reads"
        for i, tool, inf in fl:
            if tool == "get_journal_entries":
                assert not inf, (t.task_id, "journal is never faulted under cw1")
        if t.meta["faults"]:
            continue  # an earlier read may already have identified the (single-fault) world; later reads then add nothing
        first = {}
        for i, tool, inf in fl:
            first.setdefault(tool, inf)
        for tool in ("list_payments", "find_supplier_order", "list_goods_receipts"):
            if tool in first:
                assert first[tool], (t.task_id, tool, "first read after a faultable write splits the prior")


def test_never_policy_has_no_reads():
    fl, _ = flags(TASKS[0], "never")
    assert fl == []
