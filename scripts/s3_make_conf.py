"""Stage 3 confirmatory splits (PREREGISTRATION_S3), generated at registration, before any model sees them.

  s3_conf_full    300 full episodes, seed 211: the A12 training mix, with each partial fault given a ratio drawn from
                  {1/4, 1/2, 3/4} and a duplicate-delivery class added (pay only)
  s3_conf_rec     300 recovery starts, seed 212 (actionproof/stage3/tasks.py start mix)
  s3_conf_stress  100 full episodes, seed 213: duplicate delivery on pay, or lost acknowledgement then a lagging read
Every PO id and task id is disjoint from all existing ProcureSim splits (asserted in stage3.tasks.write).
    python scripts/s3_make_conf.py
"""
from __future__ import annotations

import random

import actionproof.bench.generator_proc as G
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL
from actionproof.stage3 import tasks as T

MIX = (("clean", 0.15), ("phantom", 0.20), ("partial", 0.20), ("lost_ack", 0.10), ("precommit_timeout", 0.10),
       ("never_commit", 0.10), ("dup_delivery", 0.15))


def full(split: str, n: int, seed: int, stress: bool = False) -> list:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = G.make_task(split, i, rng)
        req = t.meta["requested"]
        kind = ("dup_delivery" if rng.random() < 0.5 else "lost_ack+lag") if stress else G._pick(rng, MIX)
        t.meta.update(faults=[], never_commit=None, fault_class=kind)
        if kind in ("dup_delivery",) or kind == "lost_ack+lag":
            step = "pay"
        elif kind == "partial":
            step = rng.choice([s for s in req if s in ("pay", "receive")])
        else:
            step = rng.choice([s for s in req if s != "post"])
        if kind == "never_commit":
            t.meta["never_commit"] = STEP_TOOL[step]
        elif kind == "lost_ack+lag":
            t.meta["faults"] = [{"tool": STEP_TOOL[step], "k": 0, "kind": "lost_ack"},
                                {"tool": STEP_READ[step], "k": 0, "kind": "replica_lag"}]
        elif kind != "clean":
            f = {"tool": STEP_TOOL[step], "k": 0, "kind": kind}
            if kind == "partial":
                f["ratio"] = rng.choice(T.RATIOS)
            t.meta["faults"] = [f]
        out.append(t)
    return out


if __name__ == "__main__":
    T.write("s3_conf_full", full("s3_conf_full", 300, 211))
    T.write("s3_conf_rec", T.make_pool("s3_conf_rec", 300, 212))
    T.write("s3_conf_stress", full("s3_conf_stress", 100, 213, stress=True))
