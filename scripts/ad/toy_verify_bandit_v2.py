"""Tabular check v2 (AD-T0b): corrects AD-T0's model of AWRB.

AD-T0 (`toy_verify_bandit.py`) replaced a *read* rollout's return by its expectation over all worlds. That is a
forked-continuation Q-value, not what `actionproof/ad/worlds.py:marginalise` computes. A read reveals the world, so the
Rao-Blackwellised return of a read rollout is its own revealed-world return; only rollouts that never observe the
world get the prior mean. This file models the estimator as implemented ("rb") and keeps the AD-T0 quantity as "q"
for reference.

One aliased decision: read (cost c; after reading a faulted world, repair succeeds w.p. s) or proceed (1 if clean,
0 if faulted). World fault prob p, shared by the G rollouts of a group. VoI optimum: read iff p * s > c.

Estimators (return x scaling):
  raw/group  standard GRPO             raw/none  group-mean baseline only (Dr. GRPO advantage)
  rb/group   AWRB as registered        rb/none   AWRB returns, group-mean baseline only
  raw/batch  batch-std (LitePPO)       q/group   AD-T0's idealised full-world Q (needs forked continuations)

Outputs: final read probability per (p, c, s) and estimator; and, at fixed theta, the mean and std of the per-step
gradient (signal-to-noise) for raw/none vs rb/none.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

EST = ("raw/group", "raw/none", "raw/batch", "rb/group", "rb/none", "q/group")


def returns(est: str, a, w, rec, p, c, s):
    raw = np.where(a, np.where(w, rec * 1.0, 1.0) - c, np.where(w, 0.0, 1.0))
    kind = est.split("/")[0]
    if kind == "raw":
        return raw
    if kind == "rb":
        return np.where(a, raw, 1 - p)  # read: world revealed, own return; proceed: prior mean
    return np.where(a, (1 - p) + p * s - c, 1 - p)  # q


def grad(est, th, rng, p, c, s, G=8, prompts=4):
    pr = 1 / (1 + np.exp(-th))
    Rs, As = [], []
    for _ in range(prompts):
        a = rng.random(G) < pr
        w = np.full(G, rng.random() < p)
        rec = rng.random(G) < s
        R = returns(est, a, w, rec, p, c, s)
        Rs.append(R)
        As.append(a)
    scale = est.split("/")[1]
    g = 0.0
    allR = np.concatenate(Rs)
    for R, a in zip(Rs, As):
        A = R - R.mean()
        if scale == "group":
            A = A / (R.std() + 1e-4)
        elif scale == "batch":
            A = A / (allR.std() + 1e-4)
        g += (A * (a - pr)).sum() / G
    return g / prompts


def run(est, p, c, s, steps=3000, lr=0.5, seed=0, theta0=0.0):
    rng = np.random.default_rng(seed)
    th = theta0
    for _ in range(steps):
        th = float(np.clip(th + lr * grad(est, th, rng, p, c, s), -12, 12))
    return float(1 / (1 + np.exp(-th)))


def snr(est, th, p, c, s, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array([grad(est, th, rng, p, c, s) for _ in range(n)])
    return float(g.mean()), float(g.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grid = [(0.3, 0.05, 1.0), (0.3, 0.05, 0.6), (0.3, 0.05, 0.0), (0.15, 0.05, 0.6), (0.1, 0.02, 1.0),
            (0.45, 0.05, 0.6), (0.6, 0.05, 0.6), (0.3, 0.4, 1.0), (0.2, 0.15, 0.6)]
    rows = []
    for p, c, s in grid:
        row = {"p": p, "c": c, "s": s, "optimal": "read" if p * s > c else "proceed"}
        for est in EST:
            row[est] = round(float(np.mean([run(est, p, c, s, seed=k, theta0=th0) for k in range(a.seeds)
                                            for th0 in (0.0, 2.0)])), 3)
        for est in ("raw/none", "rb/none"):
            m, sd = snr(est, 0.0, p, c, s)
            row[f"snr@0 {est}"] = round(m / sd, 4) if sd else None
        rows.append(row)
        print(json.dumps(row))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
