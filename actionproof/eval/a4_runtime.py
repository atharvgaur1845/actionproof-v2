"""Amendment 4 runtime: env factory, inference-time middleware (PREREGISTRATION_A4 §5.1) and episode records.

Middleware only ever calls AGENT-VISIBLE read tools through the same `env.call` the model uses. It never reads
the goal spec, the fault schedule, the receipt log or the DB directly. Harness calls are tagged `by: harness`
in the trace and do not consume model turns.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict
from typing import Any

from actionproof.env.db import snapshot
from actionproof.env.tools_ecom import VERIFICATION_TOOLS, EcomEnv
from actionproof.env.verifier import verify, vsc

ENTITY_ARGS = ("order_id", "ticket_id", "sku")

# write -> reads that show its effect (seen read tools only)
RAW_MAP: dict[str, list[tuple[str, str]]] = {
    "issue_refund": [("query_ledger", "order_id")],
    "apply_store_credit": [("lookup_order", "order_id")],
    "cancel_order": [("lookup_order", "order_id")],
    "reconcile_order": [("lookup_order", "order_id")],
    "restock_item": [("check_inventory", "sku")],
    "add_audit_note": [("get_ticket", "ticket_id")],
    "close_ticket": [("get_ticket", "ticket_id")],
    "update_ticket": [("get_ticket", "ticket_id")],
    "escalate_to_human": [("get_ticket", "ticket_id")],
}


def _effect_holds(env: EcomEnv, tool: str, args: dict[str, Any]) -> bool | None:
    """Goal-agnostic predicate: is THIS write's effect already visible? None = cannot tell (always forward)."""
    o, t = args.get("order_id"), args.get("ticket_id")
    if tool == "issue_refund" and o:
        return bool(env.call("query_ledger", {"order_id": o}).get("refunds"))
    if tool == "apply_store_credit" and o:
        return bool(env.call("query_ledger", {"order_id": o}).get("credits"))
    if tool in ("cancel_order", "reconcile_order") and o:
        want = "cancelled" if tool == "cancel_order" else "refunded"
        return (env.call("lookup_order", {"order_id": o}).get("order") or {}).get("status") == want
    if tool == "add_audit_note" and t:
        return bool(env.call("get_ticket", {"ticket_id": t}).get("audit_notes"))
    if tool in ("close_ticket", "update_ticket") and t:
        want = "closed" if tool == "close_ticket" else args.get("status")
        return (env.call("get_ticket", {"ticket_id": t}).get("ticket") or {}).get("status") == want
    return None


class MiddlewareMixin:
    """mode: subset of {"raw", "vbr"}."""
    mw_modes: frozenset[str] = frozenset()
    mw_task_id: str = ""

    def _harness(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        obs = super().call(tool, args)  # type: ignore[misc]
        self.trace[-1]["by"] = "harness"  # type: ignore[attr-defined]
        return obs

    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        meta = self.META.get(tool, (None, ""))  # type: ignore[attr-defined]
        is_write = meta[1] == "write" and not self.reports  # type: ignore[attr-defined]
        if is_write and "vbr" in self.mw_modes:
            if "idempotency_key" in inspect.signature(getattr(self, tool)).parameters and not args.get("idempotency_key"):
                canon = json.dumps({k: v for k, v in args.items() if k != "idempotency_key"}, sort_keys=True)
                args["idempotency_key"] = "mw-" + hashlib.sha1(f"{self.mw_task_id}/{tool}/{canon}".encode()).hexdigest()[:16]
            ent = next((args.get(k) for k in ENTITY_ARGS if args.get(k)), None)
            seen = self.__dict__.setdefault("_mw_seen", set())
            if (tool, ent) in seen:
                holds = _effect_holds(_HarnessView(self), tool, args)
                if holds:
                    obs = {"status": "success", "note": "already applied (verified by harness); call not forwarded"}
                    self.trace.append({"idx": len(self.trace), "tool": tool, "args": args, "obs": obs,  # type: ignore[attr-defined]
                                       "by": "harness_blocked"})
                    return obs
            seen.add((tool, ent))
        obs = super().call(tool, args)  # type: ignore[misc]
        if is_write and "raw" in self.mw_modes and isinstance(args, dict):
            checks = [{"tool": rt, "result": self._harness(rt, {key: args[key]})}
                      for rt, key in RAW_MAP.get(tool, []) if args.get(key)]
            if checks:
                obs = {**obs, "automatic_verification": checks}
        return obs


class _HarnessView:
    """Lets `_effect_holds` issue reads that are tagged as harness calls."""
    def __init__(self, env: MiddlewareMixin):
        self.env = env

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        return self.env._harness(tool, args)


def make_env_a4(task, rollout_id: str, seed: int = 0, middleware: frozenset[str] = frozenset()) -> EcomEnv:
    from actionproof.rollout.agenttune_rollout import make_env
    if task.meta.get("domain") == "a4":
        from actionproof.env.tools_a4 import EcomEnvA4
        base: type = EcomEnvA4
        kw = dict(stale_status=task.stale_status, ambiguous_text=task.ambiguous_text)
        if not middleware:
            return EcomEnvA4(task.initial_state, task.schedule, rollout_id=rollout_id, seed=seed, **kw)
    else:
        if not middleware:
            return make_env(task, rollout_id, seed=seed)
        base, kw = EcomEnv, dict(stale_status=task.stale_status, ambiguous_text=task.ambiguous_text)
    cls = type(f"MW{base.__name__}", (MiddlewareMixin, base), {"mw_modes": frozenset(middleware),
                                                               "mw_task_id": task.task_id})
    return cls(task.initial_state, task.schedule, rollout_id=rollout_id, seed=seed, **kw)


def episode_record_a4(task, env: EcomEnv) -> dict[str, Any]:
    if task.meta.get("domain") == "a4":
        from actionproof.env.verifier_a4 import verify_a4 as vf
    elif task.meta.get("domain") == "bank":
        from actionproof.rollout.agenttune_rollout import episode_record
        return episode_record(task, env)
    else:
        vf = verify
    v = vf(task.goal, snapshot(env.conn), env.log, env.final_report)
    model_steps = [s for s in env.trace if not s.get("by")]
    return {
        "rollout_id": env.log.rollout_id, "task_id": task.task_id, "task_prompt": task.prompt, "vsc": vsc(v),
        "verdict": {k: val for k, val in asdict(v).items() if k != "details"},
        "violations": v.details["violations"], "trace": env.trace, "report": env.final_report,
        "n_verify": sum(s["tool"] in VERIFICATION_TOOLS for s in model_steps),
        "n_harness": sum(bool(s.get("by")) for s in env.trace),
        "n_model_calls": len(model_steps),
        "receipt_rollout_ids": sorted({r.rollout_id for r in env.log.receipts}),
        "receipts": [{"op": r.op, "entity": r.entity, "irreversible": r.irreversible, "policy_valid": r.policy_valid}
                     for r in env.log.receipts],
        "faults": task.meta.get("faults", []), "a4": task.meta.get("a4"),
        "goal": task.goal.to_json(),
        "attempts": [{"action_idx": at.action_idx, "op": at.op, "entity": at.entity, "fault": at.fault,
                      "committed": at.committed, "obs_status": at.observation.get("status")}
                     for at in env.log.attempts],
    }
