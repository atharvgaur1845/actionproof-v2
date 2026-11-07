"""A5 middleware baselines (agent-visible evidence only). Separate from the Amendment 4 middleware module, which
is hash-registered.

RAW : after every issue_refund call, the harness reads the authoritative operations log for that order and appends
      it to the observation as `automatic_verification`.
VBR : verify-before-retry + idempotency. The harness injects a deterministic idempotency key when the model omits
      one. Before forwarding a repeat refund on the same order, it reads the operations log; if a refund there is
      pending or committed, the call is not forwarded.
The harness never sees hidden state (no fault schedule, no settle-time truth, no goal spec). Harness calls are
tagged `by: harness` and are never counted as the model's evidence.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from actionproof.env.tools_a5 import EcomEnvA5


class A5Middleware(EcomEnvA5):
    mw_modes: frozenset[str] = frozenset()
    mw_task_id: str = ""

    def _harness(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        obs = super().call(tool, args)
        self.trace[-1]["by"] = "harness"
        return obs

    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        order = args.get("order_id")
        if tool == "issue_refund" and order and not self.reports:
            if "vbr" in self.mw_modes:
                if not args.get("idempotency_key"):
                    canon = json.dumps({k: v for k, v in args.items() if k != "idempotency_key"}, sort_keys=True)
                    args["idempotency_key"] = "mw-" + hashlib.sha1(f"{self.mw_task_id}/{canon}".encode()).hexdigest()[:16]
                seen = self.__dict__.setdefault("_mw_seen", set())
                if order in seen:
                    log = self._harness("list_operations", {"entity_id": order}).get("operations", [])
                    if any(o["tool"] == "issue_refund" and o["status"] in ("pending", "committed") for o in log):
                        obs = {"status": "success", "note": "a refund on this order is already pending or committed "
                                                            "(verified by harness); call not forwarded"}
                        self.trace.append({"idx": len(self.trace), "tool": tool, "args": args, "obs": obs,
                                           "by": "harness_blocked", "t": self.clock})
                        return obs
                seen.add(order)
            obs = super().call(tool, args)
            if "raw" in self.mw_modes:
                obs = {**obs, "automatic_verification": [
                    {"tool": "list_operations", "result": self._harness("list_operations", {"entity_id": order})}]}
            return obs
        return super().call(tool, args)


def make_env_a5_mw(task, rollout_id: str, seed: int = 0, middleware: frozenset[str] = frozenset()):
    from actionproof.eval.a5_runtime import make_env_a5
    if not middleware:
        return make_env_a5(task, rollout_id, seed=seed)
    cls = type("MWA5", (A5Middleware,), {"mw_modes": frozenset(middleware), "mw_task_id": task.task_id})
    return make_env_a5(task, rollout_id, seed=seed, cls=cls)
