"""Envs for RG2 (PREREGISTRATION_RG2.md): SEL1's env (transient outages) with gate v1 (the frozen RG1 gate), gate v2
(gate_v2.py) or no gate in front. Separate from sel_env.py, which RG1 addendum 1 uses unchanged."""
from __future__ import annotations

import copy
from typing import Any

from actionproof.recovery_gate.env import GateMixin, make_env_gated
from actionproof.recovery_gate.gate import Gate, Rules, intent_of
from actionproof.recovery_gate.gate_v2 import GateV2
from actionproof.sel1.env import SelEnv

GATES = {"v1": Gate, "v2": GateV2}


class GatedSelEnv2(GateMixin, SelEnv):
    pass


def make_env_rg2(task: Any, rollout_id: str, seed: int | None = None, gate: str | None = "v2") -> Any:
    m = task.meta
    txn = m.get("txn") or {}
    if txn.get("transient_unavailable"):
        faults = m.get("faults") or []
        ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
        fl = [{k: f[k] for k in ("tool", "k", "kind")} for f in faults]
        env: Any = GatedSelEnv2(task.initial_state, fl, rollout_id=rollout_id,
                                seed=m["env_seed"] if seed is None else seed, never_commit=m.get("never_commit"),
                                ratios=ratios, txn=txn)
    else:
        env = make_env_gated(task, rollout_id, seed=seed, rules=None, surface=False)
    if gate is not None:
        env.attach_gate(GATES[gate](intent_of(task), Rules(), name=env.surface_names()))
    return env


def run_policy_rg2(name: str, task: Any, gate: str | None) -> dict[str, Any]:
    """A SEL1 scripted policy (B4, B5, ALWAYS, ...) on the SEL1 env, behind gate v1/v2 or none."""
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.sel1.policy import SEL_POLICIES
    from actionproof.txn import invariants, truth
    from actionproof.txn.policies import Agent, BudgetExceeded
    env = make_env_rg2(task, f"rg2pol-{name}-{gate}-{task.task_id}", gate=gate)
    a = Agent(env, task)
    ended = "report"
    try:
        SEL_POLICIES[name](a)
    except BudgetExceeded:
        ended = "budget"
    if gate is not None:
        env.gate_finalize()
    rec = episode_record_proc(task, env)
    rec.update(policy=name, gate=env.gate_record() if gate else None, ended_by=ended, sample=-1, workflow=task.workflow)
    rec["txn"] = truth.txn_score(rec)
    rec["invariants"] = invariants.check(rec)
    rec["decision_record"] = truth.decision_record(rec)
    return copy.deepcopy(rec)
