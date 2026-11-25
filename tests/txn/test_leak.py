"""Leak test: no evaluator label, `meta.txn` content or fault name reaches the agent. Checked on everything the agent
sees for the whole dev split rendered by B4 (system prompt with tool names, signatures and docstrings, the user
prompt, every observation), and on the prompts and tool surfaces of every test-split task (no episodes run)."""
from __future__ import annotations

import inspect
import json
import re

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.txn.env import TXN_FAULT_KINDS, make_env_txn
from actionproof.txn.policies import Agent, POLICIES

LABELS = ("escalation_correct", "unavailable_reads", "quad_id", "quad_step", "reality", "template_id", "regime",
          "dup_possible", "lag_first_reads", "reorder_fields", "fault_class", "unresolvable", "delayed", "fault",
          "ok_phantom", "ok_committed", "timeout_committed", "timeout_uncommitted", "ood_alias", "ood_schema",
          "ood_workflow", "ood_combo", "ood_evidence", "ood_domain", "quad", "core", "env_seed", "never_commit",
          "replica_lag", "visibility")


def _forbidden(task) -> set[str]:
    words = {k.lower() for k in TXN_FAULT_KINDS if k != "ok"} | set(LABELS)  # "ok" is the ack status itself
    for k, v in (task.meta.get("txn") or {}).items():
        words.add(k.lower())
        if k in ("template_id", "quad_id", "reality", "group", "regime") and isinstance(v, str) and v != "standard":
            words.add(v.lower())
    words.add(str(task.meta["env_seed"]))
    return words


def _surface_text(env) -> str:
    parts = [env.adapt_prompt(PROC_NEUTRAL)]
    for t in env.tools():
        parts += [t.__name__, t.__doc__ or "", str(inspect.signature(t))]
    return "\n".join(parts)


def _assert_clean(text: str, task, where: str) -> None:
    low = text.lower()
    hits = [w for w in _forbidden(task) if re.search(rf"(?<![a-z0-9_]){re.escape(w)}(?![a-z0-9_])", low)]
    assert not hits, f"{task.task_id} {where}: {hits}"


def test_dev_split_rendered_by_b4_leaks_nothing():
    tasks = load_tasks(DATA / "txn_dev.jsonl")
    assert len(tasks) >= 100
    for t in tasks:
        env = make_env_txn(t, f"leak-{t.task_id}")
        _assert_clean(_surface_text(env), t, "system/tools")
        _assert_clean(t.prompt, t, "prompt")
        POLICIES["B4"](Agent(env, t))
        seen = [json.dumps(s["obs"]) for s in env.trace if s["by"] == "agent"]
        assert seen
        _assert_clean("\n".join(seen), t, "observations")


def test_test_split_prompts_and_surfaces_leak_nothing():
    for t in load_tasks(DATA / "txn_test.jsonl"):
        env = make_env_txn(t, f"leak-{t.task_id}")
        _assert_clean(_surface_text(env) + "\n" + t.prompt, t, "prompt/tools")


def test_prompt_independent_of_hidden_reality_within_quadruples():
    tasks = load_tasks(DATA / "txn_dev.jsonl")
    quads: dict[str, set[str]] = {}
    for t in tasks:
        q = t.meta["txn"].get("quad_id")
        if q:
            quads.setdefault(q, set()).add(t.prompt + json.dumps(t.initial_state, sort_keys=True))
    assert quads and all(len(v) == 1 for v in quads.values())
