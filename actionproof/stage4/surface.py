"""Tool-surface randomisation for GRPO training (PREREGISTRATION_SR1.md).

S4-EVAL showed that trained agents keep calling training-time tool names when the API is renamed (G1). This module
trains on several *other* API surfaces: tool names, argument names, the acknowledgement key and its ok / error / timeout
tokens. The semantics, faults, state and verifier stay ProcureSim's; the trace keeps canonical names, so the frozen
verifier and rewards score every episode unchanged (the same design as `shift.ApiShiftEnv`, which is the held-out
evaluation surface and shares no name with any surface here).

Surface choice: one surface per (task, training step), so the eight rollouts of a GRPO group share it; half of the
choices are the canonical surface. A known augmentation (domain randomisation of the tool interface), not a new method.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from typing import Any, Callable

from actionproof.integrity.sim.env import PROC_SYSTEM
from actionproof.stage3.env import S3ProcEnv
from actionproof.stage4 import shift

CANON_TOOLS = sorted(shift.INV_TOOL)
CANON_ARGS = sorted(shift.ARG_MAP)
# synonym pools; index k of every pool forms surface k. None overlaps the held-out S4 surface (asserted below).
TOOL_SYN = {
    "get_purchase_order": ["read_po", "fetch_purchase_order", "po_details", "lookup_order_doc", "po_record", "view_po"],
    "find_supplier_order": ["supplier_order_lookup", "vendor_order", "check_vendor_order", "order_at_supplier",
                            "vendor_order_state", "query_supplier_order"],
    "list_payments": ["payment_history", "show_payments", "payments_for_po", "disbursements", "paid_items",
                      "payment_records"],
    "list_goods_receipts": ["receipt_history", "show_receipts", "goods_in_records", "inbound_log", "received_items",
                            "receiving_records"],
    "get_journal_entries": ["ledger_lines", "show_journal", "accounting_entries", "gl_lines", "journal_history",
                            "posted_entries"],
    "place_supplier_order": ["submit_vendor_order", "send_order", "order_from_vendor", "raise_order", "issue_order",
                             "transmit_order"],
    "pay_supplier": ["send_payment", "make_payment", "pay_vendor", "disburse", "settle_invoice", "transfer_funds"],
    "record_goods_receipt": ["register_delivery", "receive_goods", "goods_in", "confirm_delivery", "intake_goods",
                             "post_receipt"],
    "post_journal_entry": ["write_ledger", "post_entry", "record_accounting", "add_gl_line", "journalize", "enter_ledger"],
    "refund_payment": ["void_payment", "return_payment", "claw_back", "cancel_disbursement", "undo_payment",
                       "recall_payment"],
    "escalate_to_human": ["ask_supervisor", "raise_ticket", "flag_for_review", "call_operator", "request_help",
                          "send_to_human"],
    "submit_report": ["final_answer", "close_task", "report_back", "end_task", "submit_summary", "complete_task"],
}
ARG_SYN = {
    "po_id": ["order_ref", "po_number", "po", "doc_id", "po_ref", "order_doc"],
    "amount_cents": ["cents", "value_cents", "sum_cents", "total_cents", "money_cents", "pay_cents"],
    "idempotency_key": ["dedupe_key", "op_key", "txn_key", "unique_ref", "token", "client_ref"],
    "qty": ["units", "count", "pieces", "n_units", "amount_units", "items"],
    "payment_id": ["payment", "pay_ref", "disbursement_id", "txn_id", "payment_key", "pid"],
    "reason": ["why", "comment", "explanation", "message", "details", "context"],
    "status": ["state", "verdict", "final_status", "result_code", "end_state", "disposition"],
    "summary": ["text", "description", "report_text", "recap", "overview", "account"],
}
ACK_SYN = [("ack_state", "done", "failed", "timed_out"), ("code", "OK", "ERR", "no_reply"),
           ("outcome_flag", "success", "failure", "gateway_timeout"), ("ack", "true", "false", "lost_response"),
           ("response", "COMPLETE", "ERROR", "time_limit_reached"), ("result_state", "PASS", "FAIL", "no_answer")]
K = len(ACK_SYN)


def _held_out_names() -> set[str]:
    return set(shift.TOOL_MAP) | set(shift.ARG_MAP.values()) | {"result", "ACCEPTED", "REJECTED", "deadline_exceeded"}


class Surface:
    def __init__(self, k: int):
        self.k = k
        self.tool_to_canon = {TOOL_SYN[c][k]: c for c in TOOL_SYN}
        self.canon_to_tool = {c: TOOL_SYN[c][k] for c in TOOL_SYN}
        self.arg = {c: ARG_SYN[c][k] for c in ARG_SYN}
        self.inv_arg = {v: c for c, v in self.arg.items()}
        self.ack_key, self.ok, self.err, self.timeout = ACK_SYN[k]

    def rename(self, x: Any) -> Any:
        if isinstance(x, list):
            return [self.rename(v) for v in x]
        if isinstance(x, dict):
            return {(self.arg.get(k, k) if k not in ("status", "reason", "summary") else k): self.rename(v)
                    for k, v in x.items()}
        return x

    def obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        o = {(self.ack_key if k == "status" else self.arg.get(k, k)): self.rename(v) for k, v in obs.items()}
        if self.ack_key in o:
            o[self.ack_key] = {"ok": self.ok, "error": self.err}.get(o[self.ack_key], o[self.ack_key])
        if o.get("error") == "timeout":
            o["error"], o["detail"] = self.timeout, "No response arrived; the outcome is unknown."
        return o


SURFACES = [Surface(k) for k in range(K)]
for _s in SURFACES:  # every surface is complete, internally unique, and disjoint from the held-out S4 surface
    _names = set(_s.tool_to_canon) | set(_s.arg.values()) | {_s.ack_key, _s.ok, _s.err, _s.timeout}
    assert len(_s.tool_to_canon) == len(TOOL_SYN) and len(set(_s.arg.values())) == len(ARG_SYN)
    assert not _names & _held_out_names(), _names & _held_out_names()
    assert not set(_s.tool_to_canon) & set(TOOL_SYN), "a surface name equals a canonical tool name"
    assert _s.ack_key not in _s.arg.values(), "acknowledgement key equals an argument name"


class SurfaceEnv(S3ProcEnv):
    """ProcureSim behind surface `self.surface`. The agent sees only that surface; the trace stays canonical."""
    surface: Surface

    def adapt_prompt(self, base: str) -> str:
        text = base if base.startswith(PROC_SYSTEM) else PROC_SYSTEM
        for c, new in sorted(self.surface.canon_to_tool.items(), key=lambda kv: -len(kv[0])):
            text = re.sub(rf"\b{c}\b", new, text)
        return text.replace('status "completed", "failed" or "escalated"',
                            f'{self.surface.arg["status"]} "completed", "failed" or "escalated"')

    def call(self, tool: str, args: dict[str, Any] | None = None, by: str = "agent") -> dict[str, Any]:
        S = self.surface
        if tool not in S.tool_to_canon:
            obs = {"status": "error", "error": "unknown_tool", "detail": tool}
            self.trace.append({"idx": len(self.trace), "tool": tool, "args": dict(args or {}), "obs": obs, "by": by})
            return S.obs(obs)
        old_args = {S.inv_arg.get(k, k): v for k, v in (args or {}).items()}
        return S.obs(super().call(S.tool_to_canon[tool], old_args, by))

    def tools(self) -> list[Callable[..., dict[str, Any]]]:
        S, out = self.surface, []
        for new, old in S.tool_to_canon.items():
            method = getattr(self, old)
            sig = inspect.signature(method)
            params = [p.replace(name=S.arg.get(p.name, p.name)) for p in sig.parameters.values()]
            doc = method.__doc__ or ""
            for a, b in S.arg.items():
                doc = re.sub(rf"\b{a}\b", b, doc)
            for c, n in S.canon_to_tool.items():
                doc = re.sub(rf"\b{c}\b", n, doc)

            def tool(_new=new, **kwargs: Any) -> dict[str, Any]:
                return self.call(_new, kwargs)
            tool.__name__ = tool.__qualname__ = new
            tool.__doc__ = doc
            tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
            tool.__annotations__ = {S.arg.get(k, k): v for k, v in method.__annotations__.items()}
            out.append(tool)
        return out

    def to_surface(self, tool: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self.surface.canon_to_tool[tool], {self.surface.arg.get(k, k): v for k, v in args.items()}


def make_env_surface(task: Any, rollout_id: str, seed: int | None, k: int) -> SurfaceEnv:
    """Same construction as actionproof/stage3/env.make_env_s3 (frozen), with surface k."""
    m = task.meta
    faults = m.get("faults") or []
    ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
    env = SurfaceEnv(task.initial_state, [{k_: f[k_] for k_ in ("tool", "k", "kind")} for f in faults],
                     rollout_id=rollout_id, seed=m["env_seed"] if seed is None else seed,
                     never_commit=m.get("never_commit"), ratios=ratios)
    env.surface = SURFACES[k]
    return env


def choose(task_id: str, step: int | None) -> int | None:
    """None = canonical surface (half of the choices); otherwise a surface index. Fixed per (task, step)."""
    h = int(hashlib.sha256(f"{task_id}|{step}".encode()).hexdigest(), 16)
    return None if h % 2 == 0 else (h // 2) % K


def install(rollout_module: Any) -> Callable[[Callable], Callable]:
    """Patch `make_env_s3` and `start_conversation` inside actionproof.stage3.rollout and return a decorator for its
    rollout function that records the training step (so a GRPO group shares one surface)."""
    orig_env, orig_start = rollout_module.make_env_s3, rollout_module.start_conversation
    state = {"step": None}

    def make_env(task, rid, seed=None):
        k = choose(task.task_id, state["step"])
        return orig_env(task, rid, seed=seed) if k is None else make_env_surface(task, rid, seed, k)

    def start(env, task, base_prompt):
        if not isinstance(env, SurfaceEnv):
            return orig_start(env, task, base_prompt)
        conv = [{"role": "system", "content": rollout_module.system_prompt(env, base_prompt)},
                {"role": "user", "content": task.prompt}]
        for tool, args in task.meta.get("start_prefix") or []:
            t, a = env.to_surface(tool, dict(args))
            obs = env.call(t, dict(a), by="agent")
            conv.append({"role": "assistant", "content": rollout_module.act_text(t, a)})
            conv.append({"role": "user", "content": rollout_module.TOOL_RESULT_FORMAT.format(name=t, content=json.dumps(obs))})
        return conv

    rollout_module.make_env_s3, rollout_module.start_conversation = make_env, start

    def wrap(fn: Callable) -> Callable:
        def rollout_func(prompts, trainer=None, **kw):
            state["step"] = getattr(getattr(trainer, "state", None), "global_step", None)
            return fn(prompts, trainer=trainer, **kw)
        return rollout_func
    return wrap
