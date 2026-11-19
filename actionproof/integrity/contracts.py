"""Transaction contracts for ProcureSim workflows (versioned, immutable once bound, fail-closed).

A contract states what the agent may do (grants), on which entity (the target PO), within which money limits, which
effects must hold at the end (postconditions per requested step, checked against the system of record) and which
must not happen (forbidden effects: anything on another PO, any step that was not requested). Anything not granted
is rejected. The intent text is shown to the agent and never used in a decision.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Step = Literal["order", "pay", "receive", "post"]
STEP_WRITE = {"order": "place_supplier_order", "pay": "pay_supplier", "receive": "record_goods_receipt",
              "post": "post_journal_entry"}
# Commit points: a step's write is allowed only after these earlier requested steps are verified.
PREREQUISITES: dict[str, tuple[str, ...]] = {"order": (), "receive": (), "pay": ("order", "receive"), "post": ("pay",)}
READ_TOOLS = ("get_purchase_order", "find_supplier_order", "list_payments", "list_goods_receipts", "get_journal_entries")


class Expected(BaseModel):
    model_config = ConfigDict(frozen=True)
    po_id: str
    sku: str
    qty: int = Field(gt=0)
    amount_cents: int = Field(gt=0)
    payee_account: str


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True)
    currency: str = "USD"
    per_op_max_cents: int = Field(gt=0)
    total_max_cents: int = Field(gt=0)
    approval_above_cents: int | None = None


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal["ap.contract/1"] = "ap.contract/1"
    contract_id: str
    principal: str = "northwind-procurement"
    agent_id: str = "procurement-agent"
    intent_text: str
    requested: tuple[Step, ...]
    expected: Expected
    limits: Limits
    max_write_attempts: int = Field(default=3, ge=1, le=5)
    allow_compensation: bool = True
    escalate_to: str = "ap-desk@northwind.example"

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()

    def grants(self) -> set[str]:
        g = set(READ_TOOLS) | {"escalate_to_human", "submit_report"}
        g |= {STEP_WRITE[s] for s in self.requested}
        if self.allow_compensation and "pay" in self.requested:
            g.add("refund_payment")
        return g

    @classmethod
    def for_task(cls, task: Any) -> "Contract":
        """The contract an operator would bind for a ProcureSim task (derived from the approved PO, not from faults)."""
        init = task.initial_state
        po = next(p for p in init["pos"] if p["po_id"] == task.meta["po_id"])
        sup = next(s for s in init["suppliers"] if s["supplier_id"] == po["supplier_id"])
        return cls(contract_id=f"C-{task.task_id}", intent_text=task.prompt, requested=tuple(task.meta["requested"]),
                   expected=Expected(po_id=po["po_id"], sku=po["sku"], qty=po["qty"], amount_cents=po["amount_cents"],
                                     payee_account=sup["payee_account"]),
                   limits=Limits(per_op_max_cents=po["amount_cents"], total_max_cents=po["amount_cents"]))


def json_schema() -> dict[str, Any]:
    return Contract.model_json_schema()
