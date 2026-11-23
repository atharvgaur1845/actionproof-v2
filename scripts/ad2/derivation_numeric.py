"""Numerical demonstration for GRPO_FAILURE_DERIVATION.md (one aliased verify-or-skip decision, Bernoulli policy).

Per rollout: read r ~ Bern(rho). Hidden world z ~ Bern(p) (fault), shared by the group (condition A) or i.i.d. per
rollout (condition B). A read in a faulted world reveals the fault; the repair succeeds with probability s, per rollout.
Optional exogenous outcome noise: the task fails anyway with probability q0, independently of everything (a source of
within-group variance unrelated to verification).
  R = V*succ - c*r - Lf*(1-succ)
Expected update on the read logit: E[ sum_i A_i (r_i - rho) ] per group, estimated over n groups, for
  grpo    A = (R - mean) / (std_Bessel + 1e-4)
  nostd   A = R - mean
  gdpo    A = sum_k (R_k - mean_k)/(std_k + 1e-4) over components (U, -cost, -fail), then batch-normalised
The sign tells which way GRPO pushes the read probability; the VOI rule says read iff p*s*(V+Lf) > c (q0 = 0) or
p*s*(1-q0)*(V+Lf) > c.

    python scripts/ad2/derivation_numeric.py --out results/ad2/derivation_numeric.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def groups(rng, n, G, p, s, c, rho, cond, q0=0.0, V=1.0, Lf=0.0):
    r = rng.random((n, G)) < rho
    z = (rng.random((n, 1)) < p) * np.ones((1, G), bool) if cond == "A" else rng.random((n, G)) < p
    fixed = rng.random((n, G)) < s
    noise_fail = rng.random((n, G)) < q0
    succ = (~z | (r & fixed)) & ~noise_fail
    U = V * succ
    cost = c * r
    fail = Lf * (~succ)
    return r, U, cost, fail


def update(est, r, U, cost, fail, rho):
    R = U - cost - fail
    if est == "grpo":
        A = (R - R.mean(1, keepdims=True)) / (R.std(1, ddof=1, keepdims=True) + 1e-4)
    elif est == "nostd":
        A = R - R.mean(1, keepdims=True)
    elif est == "gdpo":
        A = np.zeros_like(R)
        for x, sg in ((U, 1), (cost, -1), (fail, -1)):
            A += sg * (x - x.mean(1, keepdims=True)) / (x.std(1, ddof=1, keepdims=True) + 1e-4)
        A = (A - A.mean()) / (A.std() + 1e-4)
    return float((A * (r - rho)).sum(1).mean())


def threshold(est, cond, s, c, rho=0.3, G=8, q0=0.0, Lf=0.0, n=40000, seed=0):
    """Smallest p on a grid at which the expected update on the read logit is positive (None if never)."""
    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(0.02, 1.0, 0.02), 2)
    vals = []
    for p in grid:
        u = update(est, *groups(rng, n, G, p, s, c, rho, cond, q0, Lf=Lf), rho)
        vals.append(u)
    pos = [p for p, u in zip(grid, vals) if u > 0]
    return (float(pos[0]) if pos else None), [round(v, 4) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"rows": []}
    for est in ("grpo", "nostd", "gdpo"):
        for cond in ("A", "B"):
            for s in (1.0, 0.6):
                for c in (0.0, 0.01, 0.05, 0.2):
                    for q0 in (0.0, 0.3):
                        for Lf in (0.0, 1.0):
                            if est == "gdpo" and (cond == "B" or q0 > 0 or Lf > 0):
                                continue
                            t, _ = threshold(est, cond, s, c, q0=q0, Lf=Lf)
                            voi_t = c / (s * (1 - q0) * (1 + Lf)) if c > 0 else 0.0
                            row = {"est": est, "cond": cond, "s": s, "c": c, "q0": q0, "Lf": Lf,
                                   "learned_threshold_p": t, "voi_threshold_p": round(voi_t, 4)}
                            res["rows"].append(row)
                            print(row, flush=True)
    # warm-start dependence of grpo's condition-A threshold
    res["rho_dependence"] = {str(rho): threshold("grpo", "A", 1.0, 0.05, rho=rho)[0] for rho in (0.05, 0.3, 0.6, 0.9)}
    print("rho dependence (grpo, A, s=1, c=0.05):", res["rho_dependence"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
