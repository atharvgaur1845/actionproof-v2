"""Scripted verification-rule policies for the costed-verification testbed (CPU baselines; never trained on).

Each policy writes the requested steps in order with the oracle's argument and key conventions. After each write it
decides whether to read the step's record by a fixed rule; when it reads and the record is wrong it repairs exactly as
the Stage 3 oracle does (actionproof/stage3/scripted.py:_repair), then continues. A policy that does not read moves
on. It reports "completed" if every read it made showed the step held (or it did not read), else "failed".

Rules (the "verification policy" baselines of the assignment):
  never           never read
  always          read after every write
  random_q        read after each write with probability q (seeded per task)
  timeout_only    read only after a visible timeout acknowledgement
  skip_post       read after every write except the journal entry (zero value under the CW prior)
  pay_only        read only after payments (+ after timeouts)
"""
from __future__ import annotations

import random
from typing import Any, Callable

from actionproof.integrity.sim.agents import _args, _po
from actionproof.integrity.sim.env import STEP_TOOL
from actionproof.stage3.scripted import _repair


def rule_policy(rule: Callable[[str, dict[str, Any], random.Random], bool]) -> Callable[[Any, Any], None]:
    def policy(env, task) -> None:
        po = _po(task)
        rng = random.Random(f"{task.task_id}:rule")
        ok = True
        for step in task.meta["requested"]:
            key = f"{step}-{po['po_id']}"
            obs = env.call(STEP_TOOL[step], _args(step, po, key))
            if rule(step, obs, rng):
                if not _repair(env, po, step, key, first_written=True):
                    ok = False
                    break
            elif obs.get("status") != "ok":  # blind retry with the same key (the naive convention)
                env.call(STEP_TOOL[step], _args(step, po, key))
        env.call("submit_report", {"status": "completed" if ok else "failed", "summary": "done" if ok else "failed"})
    return policy


def _timeout(obs: dict[str, Any]) -> bool:
    return obs.get("status") != "ok"


RULES: dict[str, Callable[[str, dict[str, Any], random.Random], bool]] = {
    "never": lambda s, o, r: False,
    "always": lambda s, o, r: True,
    "random_0.25": lambda s, o, r: r.random() < 0.25,
    "random_0.5": lambda s, o, r: r.random() < 0.5,
    "random_0.75": lambda s, o, r: r.random() < 0.75,
    "timeout_only": lambda s, o, r: _timeout(o),
    "skip_post": lambda s, o, r: s != "post" or _timeout(o),
    "pay_only": lambda s, o, r: s == "pay" or _timeout(o),
}
POLICIES = {k: rule_policy(v) for k, v in RULES.items()}
