"""Scripted verification-rule baselines on a CW split (CPU, deterministic). For each rule: VSC, R_V, reads per
episode and J_c = R_V - c * reads on the split's actual worlds, plus the prior-expected values (each task's every
prior world enumerated), which remove fault-sampling noise.

    python scripts/ad/scripted_frontier.py --split cw_dev --out results/ad/scripted/cw_dev_frontier.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from actionproof.ad.evaluate import verify_after
from actionproof.ad.rewards import COST
from actionproof.ad.scripted import POLICIES
from actionproof.ad.worlds import make_env_cw, world_prior
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.integrity.sim.runtime import episode_record_proc


def one(name, t):
    env = make_env_cw(t, "s")
    POLICIES[name](env, t)
    return episode_record_proc(t, env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="cw_dev")
    ap.add_argument("--cost", type=float, default=COST)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = load_tasks(DATA / f"{a.split}.jsonl")
    out = {}
    for name in POLICIES:
        recs = [one(name, t) for t in tasks]
        n = len(recs)
        vsc = sum(r["score"]["vsc"] for r in recs) / n
        rv = sum(r["score"]["r_v"] for r in recs) / n
        reads = sum(r["n_verify"] for r in recs) / n
        dup = sum(r["score"]["duplicate_irreversible"] for r in recs) / n
        e_rv = e_reads = 0.0
        for t in tasks:
            for fs, p in world_prior(t.meta["requested"]):
                tw = copy.deepcopy(t)
                tw.meta["faults"] = fs
                r = one(name, tw)
                e_rv += p * r["score"]["r_v"] / n
                e_reads += p * r["n_verify"] / n
        va = {}
        for r in recs:
            for s, (x, y) in verify_after(r).items():
                va.setdefault(s, [0, 0])
                va[s][0] += x
                va[s][1] += y
        out[name] = {"vsc": vsc, "r_v": rv, "reads": reads, "j_c": rv - a.cost * reads, "duplicate": dup,
                     "prior_r_v": e_rv, "prior_reads": e_reads, "prior_j_c": e_rv - a.cost * e_reads,
                     "verify_after": {s: x / y for s, (x, y) in va.items() if y}}
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out[name].items()}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"split": a.split, "cost": a.cost, "rules": out}, indent=1))


if __name__ == "__main__":
    main()
