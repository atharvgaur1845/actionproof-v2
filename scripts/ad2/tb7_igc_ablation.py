"""V2-TB7: where does igc's gain come from? (registered in docs/research/algorithm_discovery_v2/V2TB7_IGC_ABLATION.md)

Variants of the information gate, implemented by wrapping actionproof.ad2.tabular.policy.informative (no edit of
the TB-registered modules):
  igc_full   the registered igc (V1, V2 and V2c decisions gated when uninformative)
  igc_v1     gate only V1 (first verification after a write: e.g. the never-faulted op)
  igc_v2     gate only re-verification decisions (V2 after a repair, V2c after a clean read)
V2 after a repair is labelled uninformative in the testbed because the replay rows hold the rollout's own repair-
success bit fixed, although the agent cannot observe it (audit of the TB implementation; see FAILURE_ANALYSIS V-F9).
Settings: the 12 TB2 settings, budgets x1 and x4, 5 seeds; nostd at x1 and x4 from TB2 is the reference.

    python scripts/ad2/tb7_igc_ablation.py [--workers 14]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/ad2/tabular/TB7.jsonl"


def runs():
    out = []
    for mode in ("single", "multi"):
        for c in (0.02, 0.05, 0.1):
            for L in (0.0, 1.0):
                for seed in range(5):
                    for var in ("igc_full", "igc_v1", "igc_v2"):
                        for mult in (1, 4):
                            out.append({"id": f"TB7|{mode}|{c}|{L}|{var}|x{mult}|{seed}", "mode": mode, "c": c, "L": L,
                                        "var": var, "mult": mult, "seed": seed})
    return out


def run_one(r):
    import actionproof.ad2.tabular.policy as P
    from actionproof.ad2.tabular import envs as E
    from actionproof.ad2.tabular.train import train
    orig = P.informative

    def gate(dtype, j, st):
        if r["var"] == "igc_v1" and dtype in ("V2", "V2c"):
            return True   # treated as informative: not gated
        if r["var"] == "igc_v2" and dtype == "V1":
            return True
        return orig(dtype, j, st)
    P.informative = gate
    tasks = E.rich(r["mode"], r["c"], r["L"])
    t = time.time()
    res = train({"tasks": tasks, "est": "igc", "seed": r["seed"], "steps": 200 * r["mult"], "n_mc": 1000,
                 "eval_tasks": tasks})
    P.informative = orig
    return {"id": r["id"], **{k: res[k] for k in ("episodes_used", "eval")}, "seconds": time.time() - t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text().splitlines() if l.strip()}
    todo = [r for r in runs() if r["id"] not in done]
    print(f"TB7: {len(todo)} runs", flush=True)
    with Pool(a.workers) as pool, OUT.open("a") as f:
        for res in pool.imap_unordered(run_one, todo):
            f.write(json.dumps(res, default=float) + "\n")
            f.flush()
    print("TB7 done", flush=True)


if __name__ == "__main__":
    main()
