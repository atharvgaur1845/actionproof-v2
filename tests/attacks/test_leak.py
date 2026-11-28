"""No evaluator label reaches the agent: system prompt and tool surface, the user prompt, and every observation of the
dev split rendered by OBD with and without the gate (the policy that reads the most). Held-out tasks: prompts and
tool surfaces only (no episodes run on the held-out split)."""
from __future__ import annotations

import inspect
import json
import re

from actionproof.attacks import spec as S
from actionproof.attacks.backend import make_env_attack
from actionproof.attacks.generate import load_split, make_split
from actionproof.attacks.policies import POLICIES, policy_view
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.recovery_gate.gate import Rules
from actionproof.txn.policies import Agent, BudgetExceeded

LABELS = {"attack", "twin", "control", "heldout", "held-out", "template", "variant", "suite", "atk1", "denies",
          "confirms", "compound", "stale_plain", "po_note", "ack_note", "record_note", "audit_note", "error_ack",
          "audit_entry", "ack_pointer", "lost_ack", "precommit_timeout", "phantom", "env_seed"}
LABELS |= {f.lower() for f in S.FAMILIES + S.HELDOUT_ONLY_FAMILIES}


def _forbidden(t) -> set[str]:
    w = set(LABELS) | {str(t.meta["env_seed"]), t.meta["attack"]["pair_id"].lower()}
    for c in t.meta["attack"]["components"] + (t.meta["attack"].get("twin_of") or []):
        w.add(c["template_id"].lower())
    return w


def _clean(text: str, t, where: str) -> None:
    low = text.lower()
    hits = [w for w in _forbidden(t) if re.search(rf"(?<![a-z0-9_]){re.escape(w)}(?![a-z0-9_])", low)]
    assert not hits, f"{t.task_id} {where}: {hits}"


def _surface(env) -> str:
    parts = [env.adapt_prompt(PROC_NEUTRAL)]
    for f in env.tools():
        parts += [f.__name__, f.__doc__ or "", str(inspect.signature(f))]
    return "\n".join(parts)


def test_dev_split_rendered_by_obedient_leaks_nothing():
    for t in load_split("attack_dev"):
        for gate in (False, True):
            env = make_env_attack(t, "leak", rules=Rules() if gate else None)
            _clean(_surface(env), t, "surface")
            _clean(t.prompt, t, "prompt")
            try:
                POLICIES["OBD"](Agent(env, policy_view(t)))
            except BudgetExceeded:
                pass
            seen = [json.dumps(s["obs"], ensure_ascii=False) for s in env.trace if s.get("by", "agent") == "agent"]
            assert seen
            _clean("\n".join(seen), t, "observations")


def test_heldout_prompts_and_surfaces_leak_nothing():
    for t in make_split("attack_heldout"):
        env = make_env_attack(t, "leak")
        _clean(_surface(env) + "\n" + t.prompt, t, "prompt/surface")


def test_policy_view_has_no_attack_metadata():
    t = load_split("attack_dev")[0]
    v = policy_view(t)
    assert set(v.meta) == {"po_id", "requested"}
