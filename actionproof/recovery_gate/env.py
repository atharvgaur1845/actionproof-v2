"""The gate in front of the TXN tools. `GateMixin` sits below `TxnSurfaceEnv` in the MRO, so the gate always works on
canonical tool and argument names, and its replies are renamed for the renamed surfaces like any tool reply.

The gate holds no reference to the env: `GateMixin.call` hands it the agent's proposed call and a callback that
executes a call. Calls the gate makes itself are ordinary agent-side calls (they reach the services, count in the
cost metrics and advance late commits); their trace entries carry `"gate": <tag>`. A blocked call never reaches a
service: it is not in the trace, it does not advance the fault schedule, and it is recorded in `gate_log`.
"""
from __future__ import annotations

import copy
from typing import Any

from actionproof.recovery_gate.gate import Gate, Rules, intent_of
from actionproof.txn.env import TxnEnv, TxnSurfaceEnv


class GateMixin(TxnEnv):
    gate: Gate | None = None

    def attach_gate(self, gate: Gate) -> None:
        self.gate = gate

    def surface_names(self):
        """A plain function over copies of the surface's name maps (the gate gets no reference to the env)."""
        c2t = dict(getattr(self, "c2t", None) or {})
        arg = dict(getattr(self, "arg", None) or {})
        return lambda name: c2t.get(name) or arg.get(name) or name

    def call(self, tool: str, args: dict[str, Any] | None = None, by: str = "agent") -> dict[str, Any]:
        if by != "agent" or self.gate is None:
            return super().call(tool, args, by)
        return self.gate.handle(tool, dict(args or {}), self._gate_exec)

    def _gate_exec(self, tool: str, args: dict[str, Any], tag: str) -> dict[str, Any]:
        n = len(self.trace)
        obs = super().call(tool, copy.deepcopy(args), "agent")
        if tag != "agent":
            for e in self.trace[n:]:
                if e.get("by", "agent") == "agent":
                    e["gate"] = tag
        return obs

    def gate_finalize(self) -> None:
        if self.gate is not None and not self.reports:
            self.gate.finalize(self._gate_exec)

    def gate_record(self) -> dict[str, Any]:
        g = self.gate
        if g is None:
            return {}
        return {"rules": vars(g.rules) if not hasattr(g.rules, "__dataclass_fields__") else
                {k: getattr(g.rules, k) for k in g.rules.__dataclass_fields__},
                "log": copy.deepcopy(g.log), "rejections": g.rejections,
                "gate_calls": sum(1 for e in self.trace if e.get("gate"))}


class GatedTxnEnv(GateMixin, TxnEnv):
    pass


class GatedSurfaceEnv(TxnSurfaceEnv, GateMixin):
    pass


def make_env_gated(task: Any, rollout_id: str, seed: int | None = None, rules: Rules | None = Rules(),
                   surface: bool = True) -> TxnEnv:
    """make_env_txn's construction (actionproof/txn/env.py), with the gate in front when `rules` is not None."""
    m = task.meta
    faults = m.get("faults") or []
    ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
    txn = m.get("txn") or {}
    kw = dict(rollout_id=rollout_id, seed=m["env_seed"] if seed is None else seed,
              never_commit=m.get("never_commit"), ratios=ratios, txn=txn)
    fl = [{k: f[k] for k in ("tool", "k", "kind")} for f in faults]
    if surface and txn.get("surface"):
        env: GateMixin = GatedSurfaceEnv(task.initial_state, fl, surface=txn["surface"], **kw)
    else:
        env = GatedTxnEnv(task.initial_state, fl, **kw)
    if rules is not None:
        env.attach_gate(Gate(intent_of(task), rules, name=env.surface_names()))
    return env


def run_policy_gated(name: str, task: Any, rules: Rules = Rules()) -> dict[str, Any]:
    """A scripted baseline (actionproof/txn/policies.py) with the gate in front, scored like any episode."""
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.txn import invariants, policies, truth
    env = make_env_gated(task, f"rg1pol-{name}-{task.task_id}", rules=rules, surface=False)
    a = policies.Agent(env, task)
    ended = "report"
    try:
        policies.POLICIES[name](a)
    except policies.BudgetExceeded:
        ended = "budget"
    env.gate_finalize()
    rec = episode_record_proc(task, env)
    rec.update(policy=f"{name}+gate", ended_by=ended, sample=-1, workflow=task.workflow, gate=env.gate_record())
    rec["txn"] = truth.txn_score(rec)
    rec["invariants"] = invariants.check(rec)
    rec["decision_record"] = truth.decision_record(rec)
    return rec


__all__ = ["GateMixin", "GatedTxnEnv", "GatedSurfaceEnv", "make_env_gated", "run_policy_gated"]
