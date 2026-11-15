"""ActionProof-Bench environment v2 (versioned; v1 in tools_ecom.py is unchanged and stays hash-frozen).

Two changes from v1, both from external review (2026-11-15):
1. Idempotency is scoped to (operation, entity, key), not the raw key string. In v1 a key reused across operations
   replayed the other operation's result (disclosed deviation D-ENV-1).
2. A timeout can come before the commit. Calls listed in `precommit_timeouts` time out without committing (the agent
   sees the same error as for a timeout after the commit, so only a read can tell them apart).
A task opts in with meta {"env_version": 2, "precommit_timeouts": [[tool, k], ...]}.
"""
from __future__ import annotations

from typing import Any

from actionproof.env.faults import COMMITS, Fault
from actionproof.env.ledger import Receipt
from actionproof.env.tools_ecom import Attempt, EcomEnv, Delta, apply_deltas, entity_hash, snapshot


class EcomEnvV2(EcomEnv):
    def __init__(self, *args: Any, precommit_timeouts: list | None = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.precommit = {tuple(x) for x in (precommit_timeouts or [])}

    def _write(self, entity_key: str, entity: str, deltas: list[Delta], success_obs: dict[str, Any], *,
               irreversible: bool, policy_valid: bool = True, idempotency_key: str | None = None) -> dict[str, Any]:
        action_idx, tool, k = self._ctx
        scoped = (tool, entity, idempotency_key) if idempotency_key else None
        if scoped and scoped in self.idempotency:
            return self.idempotency[scoped]  # exactly-once per (operation, entity, key)

        fault = self.schedule.fault_for(tool, k)
        if fault is Fault.AMBIGUOUS:
            commits = self.schedule.ambiguous_commits[(tool, k)]
        elif fault is Fault.TIMEOUT and (tool, k) in self.precommit:
            commits = False  # timeout before commit
        else:
            commits = COMMITS.get(fault, True) if fault else True
        applied = deltas[:1] if fault is Fault.PARTIAL else deltas

        pre = entity_hash(snapshot(self.conn), entity_key, entity)
        if commits:
            apply_deltas(self.conn, applied)
            post = entity_hash(snapshot(self.conn), entity_key, entity)
            self.log.append(Receipt(
                rollout_id=self.log.rollout_id, action_idx=action_idx, op=tool, entity=entity,
                pre_hash=pre, post_hash=post, txn_id=self._next_id("TX"),
                policy_valid=policy_valid, idempotency_key=idempotency_key,
                deltas=tuple(applied), irreversible=irreversible,
            ))
            if scoped:
                self.idempotency[scoped] = success_obs

        if fault is Fault.TIMEOUT:
            obs = {"status": "error", "error": "timeout",
                   "detail": f"upstream {self.META[tool][0]} service did not respond within 30s"}
        elif fault is Fault.STALE:
            obs = self._stale_ack(success_obs)
        elif fault is Fault.AMBIGUOUS:
            obs = {"status": "accepted", "detail": self.ambiguous_text}
        else:
            obs = success_obs
        self.log.attempts.append(Attempt(action_idx, tool, entity, fault.value if fault else None, commits, obs))
        return obs
