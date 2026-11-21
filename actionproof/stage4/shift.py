"""Stage 4 benchmark shifts (docs/research/BENCHMARK_S4.md). Every shift keeps ProcureSim's semantics and verifier and
changes one surface at a time:

  language   the user request in Hindi or Hinglish (the system policy stays English)
  paraphrase a reworded English request
  api        renamed tools and arguments and a different acknowledgement format (ApiShiftEnv); the trace keeps the
             canonical names so the frozen verifier scores it unchanged
  workflow4  an unseen four-step chain: order, receive, pay, post
  combo      two faults on two different steps of one episode
"""
from __future__ import annotations

import inspect
import random
import re
from typing import Any, Callable

import actionproof.bench.generator_proc as G
from actionproof.integrity.sim.env import PROC_SYSTEM, STEP_TOOL
from actionproof.stage3.env import S3ProcEnv

ITEMS_HI = {"safety gloves": "सुरक्षा दस्ताने", "cable ties": "केबल टाई", "printer paper": "प्रिंटर पेपर",
            "drill bits": "ड्रिल बिट", "pallet wrap": "पैलेट रैप", "LED panels": "एलईडी पैनल", "hard hats": "हेलमेट",
            "toner cartridges": "टोनर कार्ट्रिज", "hex bolts": "हेक्स बोल्ट", "packing tape": "पैकिंग टेप"}


def _fields(task: Any) -> dict[str, Any]:
    po = next(p for p in task.initial_state["pos"] if p["po_id"] == task.meta["po_id"])
    sup = next(s for s in task.initial_state["suppliers"] if s["supplier_id"] == po["supplier_id"])
    m = re.search(r"\d+ x (.+?) \(SKU", task.prompt)
    return {"po": po["po_id"], "qty": po["qty"], "sku": po["sku"], "unit": po["unit_price_cents"],
            "amount": po["amount_cents"], "supplier": sup["name"], "item": m.group(1) if m else "items"}


def prompt_variant(task: Any, lang: str) -> str:
    f, wf = _fields(task), task.workflow
    if lang == "hi":
        head = (f"खरीद आदेश {f['po']} स्वीकृत है: {f['supplier']} से {f['qty']} x {ITEMS_HI.get(f['item'], f['item'])} "
                f"({f['sku']}), {f['unit']} सेंट प्रति इकाई, कुल {f['amount']} सेंट।")
        ask = {"order_prepay": "सप्लायर के पास ऑर्डर दें, सप्लायर को पूरी राशि का अग्रिम भुगतान करें, और भुगतान की जर्नल प्रविष्टि पोस्ट करें।",
               "receive_pay": f"सभी {f['qty']} इकाइयाँ आज पहुँच गईं। माल की प्राप्ति दर्ज करें, सप्लायर को पूरी राशि का भुगतान करें, और भुगतान की जर्नल प्रविष्टि पोस्ट करें।",
               "pay_post": "माल पहले ही प्राप्त हो चुका है। सप्लायर को पूरी राशि का भुगतान करें और भुगतान की जर्नल प्रविष्टि पोस्ट करें।",
               "order_receive_pay": f"सप्लायर के पास ऑर्डर दें, सभी {f['qty']} इकाइयाँ पहुँचने पर माल की प्राप्ति दर्ज करें, सप्लायर को पूरी राशि का भुगतान करें, और भुगतान की जर्नल प्रविष्टि पोस्ट करें।"}[wf]
    elif lang == "hinglish":
        head = (f"PO {f['po']} approve ho gaya hai: {f['supplier']} se {f['qty']} x {f['item']} ({f['sku']}), "
                f"{f['unit']} cents per unit, total {f['amount']} cents.")
        ask = {"order_prepay": "Supplier ke saath order place karo, supplier ko poora amount prepay karo, aur payment ki journal entry post kar do.",
               "receive_pay": f"Saare {f['qty']} units aaj aa gaye. Goods receipt record karo, supplier ko poora amount pay karo, aur payment ki journal entry post kar do.",
               "pay_post": "Goods pehle hi receive ho chuke hain. Supplier ko poora amount pay karo aur payment ki journal entry post kar do.",
               "order_receive_pay": f"Supplier ke saath order place karo, saare {f['qty']} units aane par goods receipt record karo, supplier ko poora amount pay karo, aur journal entry post kar do."}[wf]
    elif lang == "para":
        head = (f"Please handle approved purchase order {f['po']}: {f['qty']} units of {f['item']} (SKU {f['sku']}) from "
                f"{f['supplier']}, {f['unit']} cents per unit, {f['amount']} cents in total.")
        ask = {"order_prepay": "Submit the order to the supplier, pay the supplier the whole amount in advance, then book the matching journal entry.",
               "receive_pay": f"The full delivery of {f['qty']} units came in today. Log the receipt, pay the supplier the whole amount, then book the matching journal entry.",
               "pay_post": "Delivery was received earlier. Pay the supplier the whole amount, then book the matching journal entry.",
               "order_receive_pay": f"Submit the order, log the receipt once all {f['qty']} units arrive, pay the supplier the whole amount, then book the matching journal entry."}[wf]
    elif lang == "en":
        return task.prompt
    else:
        raise ValueError(lang)
    return f"{head} {ask}"


# ---------------------------------------------------------------- API shift
TOOL_MAP = {"get_po": "get_purchase_order", "supplier_order_status": "find_supplier_order",
            "payments_ledger": "list_payments", "receipts_ledger": "list_goods_receipts", "gl_entries": "get_journal_entries",
            "create_supplier_order": "place_supplier_order", "remit_payment": "pay_supplier",
            "log_receipt": "record_goods_receipt", "book_journal": "post_journal_entry", "reverse_payment": "refund_payment",
            "handoff_to_operator": "escalate_to_human", "finish": "submit_report"}
ARG_MAP = {"po_id": "purchase_order", "amount_cents": "amount_minor", "idempotency_key": "request_id", "qty": "quantity",
           "payment_id": "payment_ref", "reason": "note", "status": "outcome", "summary": "notes"}
INV_TOOL = {v: k for k, v in TOOL_MAP.items()}
INV_ARG = {v: k for k, v in ARG_MAP.items()}


def _rename(x: Any) -> Any:
    """Rename record fields to the shifted vocabulary, recursively (row fields such as an order's "status" keep their
    name; only the top-level acknowledgement status becomes "result")."""
    if isinstance(x, list):
        return [_rename(v) for v in x]
    if isinstance(x, dict):
        return {(ARG_MAP.get(k, k) if k not in ("status", "reason", "summary") else k): _rename(v) for k, v in x.items()}
    return x


def _shift_obs(obs: dict[str, Any]) -> dict[str, Any]:
    o = {("result" if k == "status" else ARG_MAP.get(k, k)): _rename(v) for k, v in obs.items()}
    if "result" in o:
        o["result"] = {"ok": "ACCEPTED", "error": "REJECTED"}.get(o["result"], o["result"])
    if o.get("error") == "timeout":
        o["error"], o["detail"] = "deadline_exceeded", "No response before the deadline; the outcome is unknown."
    return o


class ApiShiftEnv(S3ProcEnv):
    """ProcureSim behind a different API surface. The agent sees only the shifted names and acknowledgement format."""

    def adapt_prompt(self, base: str) -> str:
        text = base if base.startswith(PROC_SYSTEM) else PROC_SYSTEM
        for old, new in TOOL_MAP.items():
            text = text.replace(new, old)
        return text.replace('status "completed", "failed" or "escalated"', 'outcome "completed", "failed" or "escalated"')

    def call(self, tool: str, args: dict[str, Any] | None = None, by: str = "agent") -> dict[str, Any]:
        if tool not in TOOL_MAP:  # canonical names are not part of this API
            obs = {"status": "error", "error": "unknown_tool", "detail": tool}
            self.trace.append({"idx": len(self.trace), "tool": tool, "args": dict(args or {}), "obs": obs, "by": by})
            return _shift_obs(obs)
        old_args = {INV_ARG.get(k, k): v for k, v in (args or {}).items()}
        return _shift_obs(super().call(TOOL_MAP[tool], old_args, by))

    def tools(self) -> list[Callable[..., dict[str, Any]]]:
        out = []
        for new, old in TOOL_MAP.items():
            method = getattr(self, old)
            sig = inspect.signature(method)
            params = [p.replace(name=ARG_MAP.get(p.name, p.name)) for p in sig.parameters.values()]
            doc = method.__doc__ or ""
            for a, b in ARG_MAP.items():
                doc = re.sub(rf"\b{a}\b", b, doc)
            for a, b in TOOL_MAP.items():
                doc = doc.replace(b, a)

            def tool(_new=new, **kwargs: Any) -> dict[str, Any]:
                return self.call(_new, kwargs)
            tool.__name__ = tool.__qualname__ = new
            tool.__doc__ = doc
            tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
            tool.__annotations__ = {ARG_MAP.get(k, k): v for k, v in method.__annotations__.items()}
            out.append(tool)
        return out


def make_env_api(task: Any, rollout_id: str, seed: int | None = None) -> ApiShiftEnv:
    """Same construction as actionproof/stage3/env.make_env_s3 (frozen), with the shifted class."""
    m = task.meta
    faults = m.get("faults") or []
    ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
    return ApiShiftEnv(task.initial_state, [{k: f[k] for k in ("tool", "k", "kind")} for f in faults],
                       rollout_id=rollout_id, seed=m["env_seed"] if seed is None else seed,
                       never_commit=m.get("never_commit"), ratios=ratios)


# ---------------------------------------------------------------- unseen workflow and fault combinations
def workflow4_task(split: str, i: int, rng: random.Random) -> Any:
    t = G.make_task(split, i, rng)
    state, po, sup_name, item = G._world(rng, "order_prepay")
    t.initial_state, t.workflow = state, "order_receive_pay"
    t.meta.update(po_id=po["po_id"], requested=["order", "receive", "pay", "post"], faults=[], never_commit=None)
    t.prompt = (f"Purchase order {po['po_id']} is approved: {po['qty']} x {item} ({po['sku']}) from {sup_name} at "
                f"{po['unit_price_cents']} cents each, total {po['amount_cents']} cents. Place the order with the supplier, "
                f"record the goods receipt when all {po['qty']} units arrive, pay the supplier the full amount, and post "
                "the journal entry for the payment.")
    return t


def apply_fault(t: Any, rng: random.Random, kind: str, step: str | None = None) -> None:
    req = t.meta["requested"]
    if step is None:
        step = rng.choice([s for s in req if s in ("pay", "receive")] if kind == "partial" else
                          ["pay"] if kind == "dup_delivery" else [s for s in req if s != "post"])
    if kind == "never_commit":
        t.meta["never_commit"] = STEP_TOOL[step]
    elif kind != "clean":
        f = {"tool": STEP_TOOL[step], "k": 0, "kind": kind}
        if kind == "partial":
            f["ratio"] = rng.choice((0.25, 0.5, 0.75))
        t.meta["faults"] = t.meta.get("faults", []) + [f]


def combo_task(split: str, i: int, rng: random.Random) -> Any:
    while True:
        t = G.make_task(split, i, rng)
        if len([s for s in t.meta["requested"] if s != "post"]) >= 2:
            break
    t.meta.update(faults=[], never_commit=None)
    a, b = rng.sample([s for s in t.meta["requested"] if s != "post"], 2)
    ka = rng.choice(["phantom", "lost_ack", "precommit_timeout"] + (["partial"] if a in ("pay", "receive") else []))
    kb = rng.choice(["phantom", "lost_ack"] + (["partial", "dup_delivery"] if b == "pay" else []))
    apply_fault(t, rng, ka, a)
    apply_fault(t, rng, kb, b)
    t.meta["fault_class"] = f"combo:{ka}+{kb}"
    return t

