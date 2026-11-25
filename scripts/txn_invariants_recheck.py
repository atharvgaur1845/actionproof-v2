"""TXN1 addendum 1 (D-TXN1-1): re-apply the corrected invariant checker to every stored TXN episode.

Every record keeps its trace and attempts (with `by`), so the corrected `actionproof/txn/invariants.py` can be run on
the stored episodes without rerunning any model. Reports, per tag, the count under the checker frozen in TXN1 (the
`invariants` field written at run time) and under the corrected checker, by code. Invariants are descriptive in TXN1;
no registered hypothesis uses them.

    python scripts/txn_invariants_recheck.py --split txn_test   # -> results/txn/invariants_recheck__txn_test.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from actionproof.txn import invariants

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="txn_test")
    ap.add_argument("--res-dir", default=str(ROOT / "results/txn"))
    a = ap.parse_args(argv)
    d = Path(a.res_dir)
    out = {"deviation": "D-TXN1-1 (PREREGISTRATION_TXN1_ADDENDUM1.md)", "split": a.split, "tags": {}}
    for f in sorted(d.glob(f"*__{a.split}.jsonl")):
        tag = f.name.split("__")[0]
        frozen, fixed, n, ep_f, ep_c = Counter(), Counter(), 0, 0, 0
        for line in f.read_text().splitlines():
            r = json.loads(line)
            n += 1
            old = r.get("invariants") or []
            new = invariants.check(r)
            frozen.update(v["code"] for v in old)
            fixed.update(v["code"] for v in new)
            ep_f += bool(old)
            ep_c += bool(new)
        out["tags"][tag] = {"episodes": n, "frozen_checker": {"violations": sum(frozen.values()), "episodes_with_violation": ep_f,
                                                               "by_code": dict(frozen)},
                            "corrected_checker": {"violations": sum(fixed.values()), "episodes_with_violation": ep_c,
                                                  "by_code": dict(fixed)}}
    dst = d / f"invariants_recheck__{a.split}.json"
    dst.write_text(json.dumps(out, indent=1, sort_keys=True))
    for t, v in out["tags"].items():
        print(f"{t:14s} frozen {v['frozen_checker']['violations']:5d}  corrected {v['corrected_checker']['violations']:5d}")
    return out


if __name__ == "__main__":
    main()
