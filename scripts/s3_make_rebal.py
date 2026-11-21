"""S3 addendum 2 (PREREGISTRATION_S3_ADDENDUM2): workflow-rebalanced pools and targeted partial-payment splits.

Pools. The RSC pool (`s3_trainb_half` + `s3_rec_train_half`) holds 67 partial-payment tasks: 32 order-then-pay,
29 pay-first and 6 receive-then-pay. The rebalanced pools keep the pool size, every fault-class count, every partial
ratio and every start type. Only the workflow a partial payment sits in changes: randomly chosen order-then-pay and
pay-first partial-payment tasks are replaced in place by fresh receive-then-pay tasks carrying the same fault dict
(and, for recovery starts, the same start type), so each pool half is split as evenly as possible across workflows.
The control pool is derived from the rebalanced recovery starts exactly as `s3_make_matched.py` does.

Targeted splits. `s3_pp_full` (150 full episodes) and `s3_pp_rec` (150 recovery starts placed right after the
faulted payment): 50 tasks per workflow, a partial payment on attempt 0 of `pay_supplier`, ratios 1/4, 1/2, 3/4 in
turn; recovery starts alternate ack and read. Fresh worlds; task and PO ids disjoint from every existing split.
    python scripts/s3_make_rebal.py
"""
from __future__ import annotations

import copy
import json
import random
from collections import Counter

from actionproof.bench.generator_proc import DATA, load_tasks, make_task
from actionproof.stage3.tasks import RATIOS, prefix, seen_ids

WF = ("order_prepay", "pay_post", "receive_pay")
PAY = "pay_supplier"


def is_pp(t) -> bool:
    f = t.meta.get("faults") or []
    return bool(f) and f[0]["kind"] == "partial" and f[0]["tool"] == PAY


class Fresh:
    """Fresh tasks of one workflow from one generator stream (ids `<name>-<i>`, gaps where the workflow differs)."""

    def __init__(self, name: str, seed: int):
        self.name, self.rng, self.i = name, random.Random(seed), 0

    def take(self, workflow: str):
        while True:
            t = make_task(self.name, self.i, self.rng)
            self.i += 1
            if t.workflow == workflow:
                return t


def with_fault(base, fault: dict, start_type: str | None, fault_class: str):
    t = copy.deepcopy(base)
    t.meta.update(faults=[dict(fault)], never_commit=None, fault_class=fault_class)
    if start_type:
        t.meta.update(start_prefix=prefix(t, "pay", start_type == "read"), start_type=start_type, start_step="pay")
    return t


def targets(total: int) -> dict[str, int]:
    n, r = divmod(total, 3)
    return {"receive_pay": n, "order_prepay": n + (r >= 1), "pay_post": n + (r == 2)}


def rebalance(tasks, fresh: Fresh, rng: random.Random):
    pp = [i for i, t in enumerate(tasks) if is_pp(t)]
    have = Counter(tasks[i].workflow for i in pp)
    want = targets(len(pp))
    out = list(tasks)
    for wf in ("order_prepay", "pay_post"):
        for i in sorted(rng.sample([i for i in pp if tasks[i].workflow == wf], have[wf] - want[wf])):
            old = tasks[i]
            out[i] = with_fault(fresh.take("receive_pay"), old.meta["faults"][0], old.meta.get("start_type"),
                                old.meta["fault_class"])
    assert Counter(t.workflow for t in out if is_pp(t)) == Counter(want)
    assert Counter(t.meta["fault_class"] for t in out) == Counter(t.meta["fault_class"] for t in tasks)
    return out


def targeted(name: str, seed: int, rec: bool):
    fresh, out = Fresh(name, seed), []
    for j in range(150):
        wf = WF[j % 3]
        ratio = RATIOS[(j // 3) % 3]
        st = ("ack", "read")[(j // 9) % 2] if rec else None
        out.append(with_fault(fresh.take(wf), {"tool": PAY, "k": 0, "kind": "partial", "ratio": ratio}, st,
                              f"s3:partial:{st}" if rec else "partial"))
    return out


def matched(rec_tasks):
    out = []
    for t in rec_tasks:
        m = copy.deepcopy(t)
        for k in ("start_prefix", "start_type", "start_step"):
            m.meta.pop(k, None)
        m.meta["fault_class"] = "s3m:" + t.meta["fault_class"].split(":")[1]
        m.task_id = t.task_id.replace("s3_rec_train", "s3_matched").replace("s3_rb_rec", "s3_rbm_rec")
        out.append(m)
    return out


def dump(name: str, tasks) -> None:
    (DATA / f"{name}.jsonl").write_text("".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in tasks))
    print(name, len(tasks), dict(Counter(t.workflow for t in tasks if is_pp(t))))


if __name__ == "__main__":
    tids, pos = seen_ids()
    rng = random.Random(241)
    full = rebalance(load_tasks(DATA / "s3_trainb_half.jsonl"), Fresh("s3_rb_full", 242), rng)
    rec = rebalance(load_tasks(DATA / "s3_rec_train_half.jsonl"), Fresh("s3_rb_rec", 243), rng)
    pp_full, pp_rec = targeted("s3_pp_full", 251, False), targeted("s3_pp_rec", 252, True)
    new = [t for t in full + rec if t.task_id.startswith("s3_rb_")] + pp_full + pp_rec
    new_pos = [p["po_id"] for t in new for p in t.initial_state["pos"]]
    assert len(new_pos) == len(set(new_pos)) and not set(new_pos) & pos, "PO ids collide"
    assert len({t.task_id for t in new}) == len(new) and not {t.task_id for t in new} & tids, "task ids collide"
    for pool in (full + rec, full + matched(rec)):
        assert len({t.prompt for t in pool}) == len(pool), "prompts must be unique per pool"
    dump("s3_trainb_half_rb", full)
    dump("s3_rec_train_half_rb", rec)
    dump("s3_matched_full_half_rb", matched(rec))
    dump("s3_pp_full", pp_full)
    dump("s3_pp_rec", pp_rec)
