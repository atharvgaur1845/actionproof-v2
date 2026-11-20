"""Write the care_admin OpsSim splits (spec: docs/stage2/OPSIM_SPEC.md).

data/bench/opsim/care_admin.jsonl     24 templates x 5 instances = 120 tasks (seed 122)
data/bench/opsim/care_admin_cf.jsonl  12 instances x 5 forced-evidence levels q = 60 tasks (seed 124)

Usage: python scripts/make_opsim_care_admin.py [--pilot]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from actionproof.integrity.opsim import DATA, load_tasks
from actionproof.integrity.opsim import care_admin as C


def pilot() -> None:
    tasks = load_tasks(DATA / f"{C.DOMAIN}.jsonl")
    print(f"{'policy':12s} {'class':13s} {'n':>3s} {'VSC':>5s} {'FS':>5s} {'phRB':>7s} {'corrEsc':>7s} {'dup':>4s} "
          f"{'coll':>4s}")
    for pol in C.POLICIES:
        agg: dict[str, Counter] = defaultdict(Counter)
        for t in tasks:
            s = C.run_policy(t, pol)["score"]
            a = agg[t.meta["fault_class"]]
            a["n"] += 1
            a["vsc"] += s["vsc"]
            a["fs"] += s["false_success"]
            a["ph0"] += s["phantom_readback"][0]
            a["ph1"] += s["phantom_readback"][1]
            a["ce"] += bool(s["correct_escalation"])
            a["ce_n"] += s["correct_escalation"] is not None
            a["dup"] += s["duplicate"]
            a["coll"] += s["collateral"]
        for c in list(C.TRANSFER_CLASSES) + list(C.OUT_OF_TAXONOMY):
            a = agg[c]
            ce = f"{a['ce']}/{a['ce_n']}" if a["ce_n"] else "-"
            print(f"{pol:12s} {c:13s} {a['n']:3d} {a['vsc'] / a['n']:5.2f} {a['fs'] / a['n']:5.2f} "
                  f"{a['ph0']:>3d}/{a['ph1']:<3d} {ce:>7s} {a['dup']:4d} {a['coll']:4d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="also run the scripted policies on the main split")
    a = ap.parse_args()
    print(C.write_all(DATA))
    if a.pilot:
        pilot()
