"""Tabular check of the verification decision under group-relative advantages (algorithm discovery, AD-T0).

One aliased decision: after a byte-identical write acknowledgement the policy either reads (cost c, then repairs a
silent fault with probability s) or proceeds (reward 1 if the write committed, 0 if it silently failed, fault prob p).
Policy: Bernoulli(sigmoid(theta)) on "read". G rollouts per group, `prompts` groups per step.

Estimators:
  grpo       world shared by the group (as in ActionProof: one task = one fault plan); A = (R - mean) / (std + 1e-4)
  drgrpo     world shared; A = R - mean (no std division; Dr. GRPO)
  decoupled  world drawn independently per rollout; GRPO normalisation
  awrb       world shared, but each return replaced by its expectation over the worlds that give the same observations
             (before a read every world is observation-equivalent; after a read the world is revealed, so the
             expected return of "read" is (1 - p) + p s - c); GRPO normalisation

The value-of-information optimum is: read iff p s > c (with s = 1: p > c).
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def run(est: str, p: float, c: float, s: float = 1.0, G: int = 8, steps: int = 3000, lr: float = 0.5, seed: int = 0,
        theta0: float = 0.0, prompts: int = 4) -> float:
    rng = np.random.default_rng(seed)
    th = theta0
    for _ in range(steps):
        g = 0.0
        for _ in range(prompts):
            pr = 1 / (1 + np.exp(-th))
            a = rng.random(G) < pr
            w = rng.random(G) < p if est == "decoupled" else np.full(G, rng.random() < p)
            rec = rng.random(G) < s
            R = np.where(a, np.where(w, rec * 1.0, 1.0) - c, np.where(w, 0.0, 1.0))
            if est == "awrb":
                R = np.where(a, (1 - p) + p * s - c, 1 - p)
            m, sd = R.mean(), R.std()
            A = R - m if est == "drgrpo" else (R - m) / (sd + 1e-4)
            g += (A * (a - pr)).sum() / G
        th = float(np.clip(th + lr * g / prompts, -12, 12))
    return float(1 / (1 + np.exp(-th)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grid = [(0.3, 0.05, 1.0), (0.1, 0.02, 1.0), (0.45, 0.05, 1.0), (0.55, 0.05, 1.0), (0.3, 0.0, 1.0),
            (0.3, 0.4, 1.0), (0.3, 0.05, 0.6), (0.2, 0.15, 0.6)]
    rows = []
    for p, c, s in grid:
        row = {"p": p, "c": c, "s": s, "optimal": "read" if p * s > c else "proceed"}
        for est in ("grpo", "drgrpo", "decoupled", "awrb"):
            for th0 in (0.0, 3.0):
                v = [run(est, p, c, s, seed=k, theta0=th0) for k in range(a.seeds)]
                row[f"{est}@{th0:g}"] = round(float(np.mean(v)), 3)
        rows.append(row)
        print(json.dumps(row))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
