"""The frozen Recovery Gate in front of SEL1's env (actionproof/sel1/env.py, which adds transient read outages), for
the descriptive addendum PREREGISTRATION_RG1_ADDENDUM1.md. MRO: GateMixin, then SelEnv's outage bookkeeping, then
TxnEnv, so every call the gate sends is an ordinary agent call that consumes the outage like any other."""
from __future__ import annotations

from typing import Any

from actionproof.recovery_gate.env import GateMixin
from actionproof.recovery_gate.gate import Gate, Rules, intent_of
from actionproof.sel1.env import SelEnv


class GatedSelEnv(GateMixin, SelEnv):
    pass


def make_env_gated_sel(task: Any, rollout_id: str, seed: int | None = None, rules: Rules = Rules()) -> Any:
    m = task.meta
    txn = m.get("txn") or {}
    if txn.get("transient_unavailable"):
        faults = m.get("faults") or []
        ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
        fl = [{k: f[k] for k in ("tool", "k", "kind")} for f in faults]
        env: Any = GatedSelEnv(task.initial_state, fl, rollout_id=rollout_id,
                               seed=m["env_seed"] if seed is None else seed, never_commit=m.get("never_commit"),
                               ratios=ratios, txn=txn)
    else:
        from actionproof.recovery_gate.env import make_env_gated
        env = make_env_gated(task, rollout_id, seed=seed, rules=None, surface=False)
    env.attach_gate(Gate(intent_of(task), rules, name=env.surface_names()))
    return env
