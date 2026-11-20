"""Write the OpsSim IT operations splits (docs/stage2/OPSIM_SPEC.md) and print the scripted pilot.

    uv run python scripts/make_opsim_it_ops.py            # data/bench/opsim/it_ops.jsonl (120), it_ops_cf.jsonl (60)
    uv run python scripts/make_opsim_it_ops.py --pilot    # also run oracle / naive / fixed_retry on the main split
"""
from __future__ import annotations

import argparse
from collections import Counter

from actionproof.integrity.opsim import DATA, load_tasks
from actionproof.integrity.opsim import it_ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    print(it_ops.write_data(DATA))
    tasks = load_tasks(DATA / "it_ops.jsonl")
    print(dict(sorted(Counter(t.meta["fault_class"] for t in tasks).items())))
    if a.pilot:
        res = it_ops.pilot(tasks)
        print(f"{'policy':12s} {'class':13s} {'n':>3s} {'vsc':>5s} {'false_s':>7s} {'ph_rb':>6s} {'corr_esc':>8s} "
              f"{'dup':>5s} {'coll':>5s}")
        for pol, rows in res.items():
            for c, r in rows.items():
                print(f"{pol:12s} {c:13s} {r['n']:3d} {r['vsc']:5.2f} {r['false_success']:7.2f} "
                      f"{r['phantom_readback']:6.2f} {r['correct_escalation']:8.2f} {r['duplicate']:5.2f} "
                      f"{r['collateral']:5.2f}")


if __name__ == "__main__":
    main()
