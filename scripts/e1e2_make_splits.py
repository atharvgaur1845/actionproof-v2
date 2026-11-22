"""E1/E2 (PREREGISTRATION_E1E2): fresh partial-payment splits, built the way `s3_make_rebal.py` builds the targeted
splits (same generator, same fault dict, ratios 1/4, 1/2, 3/4 in turn), with new seeds and ids disjoint from every
existing split.

  e1_pp_full  60 full episodes, 20 per workflow, partial payment on attempt 0 of pay_supplier
  e1_pp_rec   60 recovery starts right after the faulted payment, 20 per workflow, ack and read starts alternating
  e2_pp       120 read starts (prefix ends at the authoritative read of the partial payment), 40 per workflow;
              scripts/e2_probe.py builds its decision contexts from these
    python scripts/e1e2_make_splits.py
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

from actionproof.bench.generator_proc import DATA
from actionproof.stage3.tasks import RATIOS, seen_ids

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rebal", ROOT / "scripts/s3_make_rebal.py")
RB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RB)


def build(name: str, seed: int, n: int, starts: tuple[str, ...] | None):
    fresh, out = RB.Fresh(name, seed), []
    for j in range(n):
        wf = RB.WF[j % 3]
        ratio = RATIOS[(j // 3) % 3]
        st = starts[(j // 9) % len(starts)] if starts else None
        out.append(RB.with_fault(fresh.take(wf), {"tool": RB.PAY, "k": 0, "kind": "partial", "ratio": ratio}, st,
                                 f"s3:partial:{st}" if st else "partial"))
    return out


def main() -> None:
    names = ("e1_pp_full", "e1_pp_rec", "e2_pp")
    for nm in names:
        if (DATA / f"{nm}.jsonl").exists():
            raise SystemExit(f"{nm} exists; splits are built once")
    tids, pos = seen_ids()
    splits = {"e1_pp_full": build("e1_pp_full", 1101, 60, None),
              "e1_pp_rec": build("e1_pp_rec", 1102, 60, ("ack", "read")),
              "e2_pp": build("e2_pp", 1103, 120, ("read",))}
    new = [t for ts in splits.values() for t in ts]
    new_pos = [p["po_id"] for t in new for p in t.initial_state["pos"]]
    assert len(new_pos) == len(set(new_pos)) and not set(new_pos) & pos, "PO ids collide"
    assert len({t.task_id for t in new}) == len(new) and not {t.task_id for t in new} & tids, "task ids collide"
    for nm, ts in splits.items():
        assert Counter(t.workflow for t in ts) == Counter({w: len(ts) // 3 for w in RB.WF}), nm
        RB.dump(nm, ts)


if __name__ == "__main__":
    main()
