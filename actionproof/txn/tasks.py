"""TXN v1 task generator (spec: docs/txn/TXN_SPEC.md §10). `scripts/txn_make_splits.py` writes the files.

Prompts depend only on the purchase order (and, for the OOD groups, on the group's wording), never on the fault or
the regime. All evaluator labels go to `meta["txn"]`; faults stay in `meta["faults"]` (hidden, as in proc1).
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import actionproof.bench.generator_proc as G
from actionproof.bench.generator_proc import DATA, ProcTask
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL

SEEDS = {"txn_dev": 26112601, "txn_val": 26112602, "txn_test": 26112603}
SIZES = {"txn_dev": (10, 5, 0), "txn_val": (10, 5, 0), "txn_test": (25, 15, 20)}  # core per class, quads, per OOD group
TRAINED_WF = ("order_prepay", "receive_pay", "pay_post")
CORE = ("phantom", "partial", "lost_ack", "dup_delivery", "visibility_lag", "log_conflict", "late_commit", "concurrent")
CLASS_STEPS = {"phantom": ("order", "pay", "receive"), "lost_ack": ("order", "pay", "receive"),
               "log_conflict": ("order", "pay", "receive"), "partial": ("pay", "receive"),
               "dup_delivery": ("pay",), "concurrent": ("pay",), "visibility_lag": ("pay", "receive"),
               "late_commit": ("order", "pay")}
REALITIES = (("ok_phantom", "phantom"), ("ok_committed", None), ("timeout_committed", "lost_ack"),
             ("timeout_uncommitted", "precommit_timeout"))
REGIMES = ("standard", "unresolvable")
DUP_KINDS = {"lost_ack", "precommit_timeout", "log_conflict", "late_commit", "concurrent", "dup_delivery",
             "visibility_lag"}
OOD_GROUPS = ("ood_alias", "ood_schema", "ood_workflow", "ood_combo", "ood_evidence", "ood_domain")
OOD_WF = {"order_receive_pay_post": ("order", "receive", "pay", "post"), "order_pay_receive": ("order", "pay", "receive"),
          "receive_pay_nopost": ("receive", "pay")}
OOD_WORLD = {"order_receive_pay_post": "order_prepay", "order_pay_receive": "order_prepay",
             "receive_pay_nopost": "receive_pay"}
# P4's training compositions (actionproof/p4/build.py::AUG_WF); used when that module is not importable
P4_COMPOSITIONS = {("order", "receive"), ("order", "pay"), ("pay",), ("receive",), ("order",)}
COMBOS = (("order", "late_commit", "pay", "partial", "order_prepay"),
          ("order", "log_conflict", "pay", "concurrent", "order_prepay"),
          ("receive", "log_conflict", "pay", "late_commit", "receive_pay"),
          ("receive", "partial", "pay", "concurrent", "receive_pay"),
          ("order", "phantom", "pay", "log_conflict", "order_prepay"),
          ("receive", "lost_ack", "pay", "late_commit", "receive_pay"))
VENDORS = ("Kestrel Lab Supplies", "Meridian Clinical", "Northgate Instruments", "Orchid Diagnostics",
           "Pinewood Reagents", "Quarry Scientific", "Riverside Biomed", "Summit Sterile Goods")
LAB_ITEMS = ("nitrile swabs", "centrifuge tubes", "pipette tips", "specimen bags", "reagent bottles",
             "culture plates", "sterile drapes", "filter cartridges", "cryo labels", "glass slides")


def p4_compositions() -> set[tuple[str, ...]]:
    try:
        from actionproof.p4.build import AUG_WF  # type: ignore
        return {tuple(v) for v in AUG_WF.values()}
    except ImportError:
        return set(P4_COMPOSITIONS)


def existing_combos() -> set[frozenset]:
    """Fault-kind pairs of every two-fault task in p1_train_* and proc_combo."""
    out: set[frozenset] = set()
    for f in sorted(DATA.glob("p1_train_*.jsonl")) + [DATA / "proc_combo.jsonl"]:
        for line in f.read_text().splitlines():
            if line.strip():
                fl = json.loads(line)["meta"].get("faults") or []
                if len(fl) >= 2:
                    out.add(frozenset(x["kind"] for x in fl))
    return out


def _check_ood_definitions() -> None:
    p4 = p4_compositions()
    for wf, steps in OOD_WF.items():
        assert steps not in p4, f"{wf} is a P4 training composition"
        assert steps not in {tuple(v) for v in G.WORKFLOWS.values()}, f"{wf} is a trained workflow"
    seen = existing_combos()
    for a, ka, b, kb, wf in COMBOS:
        assert frozenset({ka, kb}) not in seen, f"combo {ka}+{kb} exists in a training/combo split"
        assert a in G.WORKFLOWS[wf] and b in G.WORKFLOWS[wf] and a != b


# ---------------------------------------------------------------------- worlds and prompts
def _world_vocab(rng: random.Random, workflow: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], str, str]:
    """generator_proc._world with the domain-vocabulary catalogue (same structure and distractor history)."""
    saved = G.SUPPLIERS, G.ITEMS
    G.SUPPLIERS, G.ITEMS = VENDORS, LAB_ITEMS
    try:
        return G._world(rng, workflow)
    finally:
        G.SUPPLIERS, G.ITEMS = saved


def _head(po: dict[str, Any], sup: str, item: str) -> str:
    return (f"Purchase order {po['po_id']} is approved: {po['qty']} x {item} ({po['sku']}) from {sup} at "
            f"{po['unit_price_cents']} cents each, total {po['amount_cents']} cents.")


def prompt_ood_workflow(wf: str, po: dict[str, Any], sup: str, item: str) -> str:
    ask = {"order_receive_pay_post": f"Place the order with the supplier, record the goods receipt once all {po['qty']} "
                                     "units arrive, then pay the supplier the full amount and post the journal entry for "
                                     "the payment.",
           "order_pay_receive": "Place the order with the supplier, pay the supplier the full amount, and record the "
                                f"goods receipt when all {po['qty']} units arrive. The journal entry is handled separately.",
           "receive_pay_nopost": f"All {po['qty']} units arrived today. Record the goods receipt and pay the supplier the "
                                 "full amount. The journal entry is handled separately."}[wf]
    return f"{_head(po, sup, item)} {ask}"


def prompt_domain(wf: str, po: dict[str, Any], sup: str, item: str) -> str:
    head = (f"Work order {po['po_id']} is signed off: {po['qty']} x {item} ({po['sku']}) from vendor {sup} at "
            f"{po['unit_price_cents']} cents per unit, {po['amount_cents']} cents in all.")
    ask = {"order_prepay": "Send the work order to the vendor, remit the full amount to the vendor in advance, and book "
                           "the ledger entry for the remittance.",
           "receive_pay": f"The delivery confirmation for all {po['qty']} units came in today. Log the delivery "
                          "confirmation, remit the full amount to the vendor, and book the ledger entry for the remittance.",
           "pay_post": "The delivery was confirmed earlier. Remit the full amount to the vendor and book the ledger entry "
                       "for the remittance."}[wf]
    return f"{head} {ask}"


# ---------------------------------------------------------------------- task construction
def _fault(rng: random.Random, kind: str, step: str) -> dict[str, Any]:
    f: dict[str, Any] = {"tool": STEP_TOOL[step], "k": 0, "kind": kind}
    if kind == "partial":
        f["ratio"] = rng.choice((0.25, 0.5, 0.75))
    return f


def _base(split: str, i: int, rng: random.Random, workflow: str, group: str) -> ProcTask:
    steps = OOD_WF.get(workflow) or G.WORKFLOWS[workflow]
    world_wf = OOD_WORLD.get(workflow, workflow)
    if group == "ood_domain":
        state, po, sup, item = _world_vocab(rng, world_wf)
        prompt = prompt_domain(workflow, po, sup, item)
    else:
        state, po, sup, item = G._world(rng, world_wf)
        prompt = prompt_ood_workflow(workflow, po, sup, item) if workflow in OOD_WF else G.prompt_for(workflow, po, sup, item)
    meta = {"domain": "proc", "env_version": "proc3", "po_id": po["po_id"], "requested": list(steps), "faults": [],
            "never_commit": None, "fault_class": None, "env_seed": rng.randrange(2**31),
            "txn": {"group": group, "regime": "standard", "unavailable_reads": [], "escalation_correct": False,
                    "surface": None, "lag_first_reads": 0, "reorder_fields": False}}
    return ProcTask(f"{split}-{i:04d}", split, workflow, prompt, state, meta)


def _finish(t: ProcTask, cls: str) -> ProcTask:
    tx = t.meta["txn"]
    kinds = {f["kind"] for f in t.meta["faults"]}
    if "visibility_lag" in kinds:
        tx["regime"] = "delayed"
    tx["dup_possible"] = bool(kinds & DUP_KINDS)
    t.meta["fault_class"] = cls
    tx["template_id"] = f"{tx['group']}/{t.workflow}/{cls}/{tx['regime']}"
    return t


def core_task(split: str, i: int, rng: random.Random, cls: str, workflow: str, group: str = "core") -> ProcTask:
    t = _base(split, i, rng, workflow, group)
    step = rng.choice([s for s in CLASS_STEPS[cls] if s in t.meta["requested"]])
    t.meta["faults"] = [_fault(rng, cls, step)]
    return _finish(t, cls)


def quad_tasks(split: str, i0: int, rng: random.Random, workflow: str, qi: int) -> list[ProcTask]:
    base = _base(split, i0, rng, workflow, "quad")
    step = rng.choice([s for s in base.meta["requested"] if s != "post"])
    out = []
    for regime in REGIMES:
        for reality, kind in REALITIES:
            t = copy.deepcopy(base)
            t.task_id = f"{split}-{i0 + len(out):04d}"
            tx = t.meta["txn"]
            t.meta["faults"] = [_fault(rng, kind, step)] if kind else []
            tx.update(quad_id=f"{split}-q{qi:02d}", reality=reality, quad_step=step)
            if regime == "unresolvable":
                tx.update(regime="unresolvable", unavailable_reads=[STEP_READ[step], "get_audit_log"],
                          escalation_correct=True)
            out.append(_finish(t, f"quad:{reality}"))
    return out


def ood_task(split: str, i: int, rng: random.Random, group: str, j: int) -> ProcTask:
    if group == "ood_workflow":
        wf = sorted(OOD_WF)[j % len(OOD_WF)]
        return core_task(split, i, rng, CORE[j % len(CORE)], wf, group)
    if group == "ood_combo":
        a, ka, b, kb, wf = COMBOS[j % len(COMBOS)]
        t = _base(split, i, rng, wf, group)
        t.meta["faults"] = [_fault(rng, ka, a), _fault(rng, kb, b)]
        return _finish(t, f"combo:{a}.{ka}+{b}.{kb}")
    t = core_task(split, i, rng, CORE[j % len(CORE)], TRAINED_WF[j % len(TRAINED_WF)], group)
    tx = t.meta["txn"]
    if group == "ood_alias":
        tx["surface"] = "alias"
    elif group == "ood_schema":
        tx["surface"] = "schema"
    elif group == "ood_evidence":
        tx.update(reorder_fields=True, lag_first_reads=2)
    return t


def make_split(split: str) -> list[ProcTask]:
    _check_ood_definitions()
    n_core, n_quad, n_ood = SIZES[split]
    rng = random.Random(SEEDS[split])
    out: list[ProcTask] = []
    for cls in CORE:
        for j in range(n_core):
            out.append(core_task(split, len(out), rng, cls, TRAINED_WF[j % len(TRAINED_WF)]))
    for q in range(n_quad):
        out += quad_tasks(split, len(out), rng, TRAINED_WF[q % len(TRAINED_WF)], q)
    for g in OOD_GROUPS if n_ood else ():
        for j in range(n_ood):
            out.append(ood_task(split, len(out), rng, g, j))
    return out


# ---------------------------------------------------------------------- disjointness
def existing(exclude_prefix: str = "txn_") -> tuple[set[str], set[str], set[str]]:
    tids, pos, prompts = set(), set(), set()
    for f in sorted(DATA.glob("*.jsonl")):
        if f.name.startswith(exclude_prefix):
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                tids.add(d.get("task_id"))
                prompts.add(d.get("prompt"))
                pos |= {p["po_id"] for p in d.get("initial_state", {}).get("pos", [])}
    return tids, pos, prompts


def check_disjoint(splits: dict[str, list[ProcTask]]) -> None:
    tids, pos, prompts = existing()
    seen_tid: set[str] = set()
    po_owner: dict[str, str] = {}
    prompt_owner: dict[str, str] = {}
    for tasks in splits.values():
        for t in tasks:
            owner = t.meta["txn"].get("quad_id") or t.task_id
            assert t.task_id not in tids and t.task_id not in seen_tid, f"task id {t.task_id} collides"
            seen_tid.add(t.task_id)
            assert t.prompt not in prompts, f"{t.task_id}: prompt text collides with an existing split"
            assert prompt_owner.setdefault(t.prompt, owner) == owner, f"{t.task_id}: prompt shared across tasks"
            for p in t.initial_state["pos"]:
                assert p["po_id"] not in pos, f"{t.task_id}: PO id {p['po_id']} collides with an existing split"
                assert po_owner.setdefault(p["po_id"], owner) == owner, f"{t.task_id}: PO id {p['po_id']} reused"


def task_line(t: ProcTask) -> str:
    return json.dumps(t.to_json(), sort_keys=True) + "\n"


def write_splits(out: Path = DATA) -> dict[str, Any]:
    import hashlib
    splits = {s: make_split(s) for s in SEEDS}
    check_disjoint(splits)
    paths = {s: out / f"{s}.jsonl" for s in splits}
    man_path = out / "txn_manifest.json"
    clash = [str(p) for p in list(paths.values()) + [man_path] if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite: {clash}")
    manifest: dict[str, Any] = {"spec": "docs/txn/TXN_SPEC.md", "env_version": "proc3", "seeds": SEEDS,
                                "files": {}, "tasks": {}}
    for s, tasks in splits.items():
        text = "".join(task_line(t) for t in tasks)
        paths[s].write_text(text)
        manifest["files"][paths[s].name] = {"n": len(tasks), "sha256": hashlib.sha256(text.encode()).hexdigest()}
        for t in tasks:
            tx = t.meta["txn"]
            manifest["tasks"][t.task_id] = {"split": s, "template_id": tx["template_id"], "group": tx["group"],
                                            "regime": tx["regime"], "fault_class": t.meta["fault_class"],
                                            "quad_id": tx.get("quad_id"),
                                            "sha256": hashlib.sha256(task_line(t).encode()).hexdigest()}
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {s: manifest["files"][paths[s].name] for s in splits}
