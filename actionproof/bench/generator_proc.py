"""ProcureSim task generator (spec: docs/stage2/VERIFIER_SPEC.md §4).

Splits and seeds (Amendment 12). Prompts depend only on the purchase order, never on the fault plan.

| split                 | n   | seed | faults                                                                 |
|-----------------------|-----|------|------------------------------------------------------------------------|
| proc_demo             | 240 | 101  | training mix (SFT demonstrations only)                                 |
| proc_train            | 400 | 102  | training mix                                                           |
| proc_dev, proc_val    | 200, 100 | 103, 104 | training mix (development and model selection only)          |
| proc_conf             | 300 | 105  | training mix (confirmatory, evaluated once per frozen model)          |
| proc_combo            | 100 | 106  | two faults on two different steps (unseen combinations)               |
| proc_stress           | 100 | 107  | dup_delivery, or lost_ack followed by a lagging read replica           |
| proc_heldout_service  | 100 | 108  | single faults on post_journal_entry (accounting is never faulted elsewhere) |

Training mix: one fault on one requested step among order/pay/receive (partial only on pay/receive), by class:
clean 0.20, phantom 0.25, partial 0.10, lost_ack 0.15, precommit_timeout 0.15, never_commit 0.15.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL

SPLITS = {"proc_demo": (240, 101), "proc_train": (400, 102), "proc_dev": (200, 103), "proc_val": (100, 104),
          "proc_conf": (300, 105), "proc_combo": (100, 106), "proc_stress": (100, 107),
          "proc_heldout_service": (100, 108)}
TRAIN_MIX = (("clean", 0.20), ("phantom", 0.25), ("partial", 0.10), ("lost_ack", 0.15),
             ("precommit_timeout", 0.15), ("never_commit", 0.15))
WORKFLOWS = {"order_prepay": ("order", "pay", "post"), "receive_pay": ("receive", "pay", "post"),
             "pay_post": ("pay", "post")}
SUPPLIERS = ("Acme Tools", "Borealis Parts", "Cobalt Fasteners", "Delta Packaging", "Evergreen Paper",
             "Fulcrum Electric", "Granite Safety", "Harbor Office Supply")
ITEMS = ("safety gloves", "cable ties", "printer paper", "drill bits", "pallet wrap", "LED panels", "hard hats",
         "toner cartridges", "hex bolts", "packing tape")
DATA = Path(__file__).resolve().parents[2] / "data" / "bench" / "proc"


@dataclass
class ProcTask:
    task_id: str
    split: str
    workflow: str
    prompt: str
    initial_state: dict[str, list[dict[str, Any]]]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "split": self.split, "workflow": self.workflow, "prompt": self.prompt,
                "initial_state": self.initial_state, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ProcTask":
        return cls(d["task_id"], d["split"], d["workflow"], d["prompt"], d["initial_state"], d["meta"])


def _hex(rng: random.Random, prefix: str, width: int = 6) -> str:
    return f"{prefix}-{rng.randrange(16**width):0{width}X}"


def prompt_for(workflow: str, po: dict[str, Any], supplier: str, item: str) -> str:
    head = (f"Purchase order {po['po_id']} is approved: {po['qty']} x {item} ({po['sku']}) from {supplier} at "
            f"{po['unit_price_cents']} cents each, total {po['amount_cents']} cents.")
    ask = {"order_prepay": "Place the order with the supplier, prepay the supplier the full amount, and post the "
                           "journal entry for the payment.",
           "receive_pay": f"All {po['qty']} units arrived today. Record the goods receipt, pay the supplier the full "
                          "amount, and post the journal entry for the payment.",
           "pay_post": "The goods were already received. Pay the supplier the full amount and post the journal entry "
                       "for the payment."}[workflow]
    return f"{head} {ask}"


def _world(rng: random.Random, workflow: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], str, str]:
    names = rng.sample(SUPPLIERS, 3)
    sups = [{"supplier_id": _hex(rng, "SUP"), "name": n, "payee_account": _hex(rng, "ACCT")} for n in names]
    state: dict[str, list[dict[str, Any]]] = {"pos": [], "suppliers": sups, "escalations": [], "supplier_orders": [],
                                              "payments": [], "receipts": [], "journal": []}
    items = rng.sample(ITEMS, 5)
    n_po = rng.randint(3, 5)
    for i in range(n_po):
        sup = rng.choice(sups)
        qty, unit = rng.randint(2, 60), rng.randrange(150, 25_000, 5)
        po = {"po_id": _hex(rng, "PO", 8), "supplier_id": sup["supplier_id"], "sku": f"SKU-{rng.randrange(10**4):04d}",
              "qty": qty, "unit_price_cents": unit, "amount_cents": qty * unit, "status": "approved"}
        state["pos"].append(po)
        if i > 0 and rng.random() < 0.6:  # distractor history, so reads return realistic non-empty lists
            state["supplier_orders"].append({"order_ref": _hex(rng, "SO"), "po_id": po["po_id"], "sku": po["sku"],
                                             "qty": qty, "status": "placed"})
            if rng.random() < 0.5:
                state["payments"].append({"payment_id": _hex(rng, "PAY"), "po_id": po["po_id"],
                                          "amount_cents": po["amount_cents"], "payee_account": sup["payee_account"],
                                          "kind": "payment", "refund_of": None})
                state["journal"].append({"entry_id": _hex(rng, "JE"), "po_id": po["po_id"],
                                         "amount_cents": po["amount_cents"], "kind": "payment"})
    task_po = state["pos"][0]
    if workflow in ("receive_pay", "pay_post"):  # the order was placed earlier
        state["supplier_orders"].append({"order_ref": _hex(rng, "SO"), "po_id": task_po["po_id"],
                                         "sku": task_po["sku"], "qty": task_po["qty"], "status": "placed"})
    if workflow == "pay_post":  # and received earlier
        state["receipts"].append({"receipt_id": _hex(rng, "GR"), "po_id": task_po["po_id"], "qty": task_po["qty"],
                                  "kind": "receipt"})
    for rows in state.values():  # row order carries no information about the task PO
        rng.shuffle(rows)
    sup = next(s for s in sups if s["supplier_id"] == task_po["supplier_id"])
    return state, task_po, sup["name"], items[0]


def _pick(rng: random.Random, mix: tuple[tuple[str, float], ...]) -> str:
    x, acc = rng.random(), 0.0
    for name, p in mix:
        acc += p
        if x < acc:
            return name
    return mix[-1][0]


def _faults(rng: random.Random, split: str, requested: tuple[str, ...]) -> tuple[list[dict[str, Any]], str | None, str]:
    """(fault list, never_commit tool, fault class)."""
    trainable = [s for s in requested if s != "post"]  # accounting is held out of every training split
    if split == "proc_heldout_service":
        kind = rng.choice(["phantom", "partial", "lost_ack", "precommit_timeout", "never_commit"])
        if kind == "never_commit":
            return [], STEP_TOOL["post"], "never_commit"
        return [{"tool": STEP_TOOL["post"], "k": 0, "kind": kind}], None, kind
    if split == "proc_combo":
        a, b = rng.sample(trainable + ["post"] if len(trainable) < 2 else trainable, 2)
        ka = rng.choice(["phantom", "partial", "lost_ack", "precommit_timeout"])
        kb = rng.choice(["phantom", "partial", "lost_ack", "precommit_timeout"])
        return ([{"tool": STEP_TOOL[a], "k": 0, "kind": ka}, {"tool": STEP_TOOL[b], "k": 0, "kind": kb}], None,
                f"combo:{ka}+{kb}")
    if split == "proc_stress":
        step = rng.choice([s for s in trainable if s in ("pay", "receive")] or trainable)
        if rng.random() < 0.5:
            return [{"tool": STEP_TOOL[step], "k": 0, "kind": "dup_delivery"}], None, "dup_delivery"
        # lost ack, then the first authoritative read comes from a lagging replica
        return ([{"tool": STEP_TOOL[step], "k": 0, "kind": "lost_ack"},
                 {"tool": STEP_READ[step], "k": 0, "kind": "replica_lag"}], None, "lost_ack+replica_lag")
    kind = _pick(rng, TRAIN_MIX)
    if kind == "clean":
        return [], None, "clean"
    # partial faults only where an agent tool can repair them (pay: refund and re-pay; receive: record the rest)
    step = rng.choice([s for s in trainable if s in ("pay", "receive")] if kind == "partial" else trainable)
    if kind == "never_commit":
        return [], STEP_TOOL[step], "never_commit"
    return [{"tool": STEP_TOOL[step], "k": 0, "kind": kind}], None, kind


def make_task(split: str, i: int, rng: random.Random) -> ProcTask:
    workflow = rng.choice(sorted(WORKFLOWS))
    requested = WORKFLOWS[workflow]
    state, po, sup_name, item = _world(rng, workflow)
    faults, never, cls = _faults(rng, split, requested)
    meta = {"domain": "proc", "env_version": "proc1", "po_id": po["po_id"], "requested": list(requested),
            "faults": faults, "never_commit": never, "fault_class": cls, "env_seed": rng.randrange(2**31)}
    return ProcTask(f"{split}-{i:04d}", split, workflow, prompt_for(workflow, po, sup_name, item), state, meta)


def make_split(split: str, n: int | None = None) -> list[ProcTask]:
    size, seed = SPLITS[split]
    rng = random.Random(seed)
    return [make_task(split, i, rng) for i in range(n if n is not None else size)]


def load_tasks(path: str | Path) -> list[ProcTask]:
    return [ProcTask.from_json(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def write_splits(out: Path = DATA) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    counts = {}
    for split in SPLITS:
        tasks = make_split(split)
        for t in tasks:  # entity ids must be disjoint across splits (no world is shared)
            for p in t.initial_state["pos"]:
                if p["po_id"] in seen:
                    raise AssertionError(f"PO id {p['po_id']} reused in {split} and {seen[p['po_id']]}")
                seen[p["po_id"]] = split
        (out / f"{split}.jsonl").write_text("".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in tasks))
        counts[split] = len(tasks)
    return counts


if __name__ == "__main__":
    print(write_splits())
