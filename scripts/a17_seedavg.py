"""A17 registered positive criterion, second half (PREREGISTRATION_A17 §6): template-cluster 95% interval of the
seed-averaged phantom read-back difference (R_V - R_A and R_V - warm start). The frozen `a17_stats.py` reports per-seed
intervals only; this computes the registered seed-averaged one (deviation D-A17-1). Each bootstrap draw resamples
templates, then instances within templates, once, and applies the same draw to every seed.
    python scripts/a17_seedavg.py --out results/a17/a17_seedavg.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a17_stats", ROOT / "scripts/a17_stats.py")
a17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a17)
B = 10_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/a17/a17_seedavg.json")
    a = ap.parse_args()
    seeds = a.seeds.split(",")
    rng = random.Random(0)
    res = {}
    for d in a17.DOMAINS:
        pt = {t: a17.per_task(a17.load(t, d)) for t in ["warm_start"] + [f"{x}_s{s}" for s in seeds for x in ("rv", "ra")]}
        tasks = sorted(pt["warm_start"])
        by = defaultdict(list)
        for t in tasks:
            by[pt["warm_start"][t]["template"]].append(t)
        temps = sorted(by)

        def avg_diff(ts, other):
            ds = []
            for s in seeds:
                x = a17.pooled([pt[f"rv_s{s}"][t] for t in ts])
                y = a17.pooled([pt[other.format(s=s)][t] for t in ts])
                if x is None or y is None:
                    return None
                ds.append(x - y)
            return sum(ds) / len(ds)
        res[d] = {}
        for label, other in (("rv_minus_ra", "ra_s{s}"), ("rv_minus_warm", "warm_start")):
            est = avg_diff(tasks, other)
            reps = []
            for _ in range(B):
                ts = []
                for tp in (temps[rng.randrange(len(temps))] for _ in temps):
                    inst = by[tp]
                    ts += [inst[rng.randrange(len(inst))] for _ in inst]
                v = avg_diff(ts, other)
                if v is not None:
                    reps.append(v)
            reps.sort()
            res[d][label] = {"est": est, "ci95": [reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]],
                             "n_draws": len(reps)}
            print(d, label, round(est, 3), [round(x, 3) for x in res[d][label]["ci95"]])
    out = ROOT / a.out
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
