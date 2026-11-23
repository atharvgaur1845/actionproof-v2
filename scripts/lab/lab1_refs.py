"""Exact reference policies on the L1 settings (PREREGISTRATION_LAB1.md §3): the Bayes-optimal policy (expectimax over
the hidden fault plan, regret 0 by definition) and a scripted policy.

Scripted policy: verify after every ok acknowledgement of a write whose fault rate p > 0; repair on a fault flag;
never re-verify; verify after a timeout; after a read showing 'not applied' retry, after 'applied' proceed.

    python scripts/lab/lab1_refs.py --out results/lab/lab1_refs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.ad2.tabular import envs as E  # noqa: E402
from actionproof.ad2.tabular.env import oracle_value, policy_value  # noqa: E402

SETTINGS = [(m, c, L) for m in ("single", "multi") for c in (0.02, 0.05, 0.1) for L in (0.0, 1.0)]


def scripted(dtype, ctx):
    o = ctx["op"]
    if dtype == "V1":
        return np.array([0.0, 1.0]) if o.p > 0 else np.array([1.0, 0.0])
    if dtype == "R":
        return np.array([0.0, 1.0])
    if dtype in ("V2", "V2c"):
        return np.array([1.0, 0.0])
    if dtype == "T":
        return np.array([0.0, 1.0, 0.0])
    if dtype == "TR":
        return np.array([0.0, 1.0])
    if dtype == "TRa":
        return np.array([1.0, 0.0])
    raise ValueError(dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lab/lab1_refs.json")
    a = ap.parse_args()
    rows = []
    for mode, c, L in SETTINGS:
        for t in E.rich(mode, c, L):
            Js = oracle_value(t)
            Jsc = policy_value(t, scripted)
            rows.append({"setting": f"{mode}|{c}|{L}", "task": t.tag, "J_bayes": Js, "J_scripted": Jsc,
                         "regret_scripted": Js - Jsc})
    out = {"rows": rows, "mean_regret_scripted": float(np.mean([r["regret_scripted"] for r in rows])),
           "mean_J_bayes": float(np.mean([r["J_bayes"] for r in rows]))}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(out["mean_regret_scripted"], out["mean_J_bayes"])


if __name__ == "__main__":
    main()
