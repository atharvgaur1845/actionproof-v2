"""Append-only receipt log written by the environment, never visible to the policy.

A receipt exists only for writes that actually committed. A phantom acknowledgement
therefore leaves an `Attempt` but no `Receipt`; that asymmetry is what PCSTR's proof
coverage term (V) and the strict-success term (G) rely on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from actionproof.env.db import Delta


@dataclass(frozen=True)
class Receipt:
    rollout_id: str
    action_idx: int
    op: str
    entity: str
    pre_hash: str
    post_hash: str
    txn_id: str
    policy_valid: bool
    idempotency_key: str | None
    deltas: tuple[Delta, ...]
    irreversible: bool


@dataclass(frozen=True)
class Attempt:
    """Every write call, committed or not, with what the agent was told."""
    action_idx: int
    op: str
    entity: str
    fault: str | None
    committed: bool
    observation: dict[str, Any]


@dataclass
class ReceiptLog:
    rollout_id: str
    receipts: list[Receipt] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)

    def append(self, receipt: Receipt) -> None:
        if receipt.rollout_id != self.rollout_id:
            raise ValueError(f"receipt for {receipt.rollout_id} written to log {self.rollout_id}")
        self.receipts.append(receipt)

    def committed(self, op: str | None = None, entity: str | None = None) -> list[Receipt]:
        return [
            r for r in self.receipts
            if (op is None or r.op == op) and (entity is None or r.entity == entity)
        ]
