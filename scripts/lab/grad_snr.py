"""Bias and variance of each estimator's one-step gradient at a fixed policy (PREREGISTRATION_LAB1.md, family L2G).

For a fixed parameter vector theta, each estimator's per-step gradient on the verify logits (head V1, action 1, the
op-type one-hot entries) is measured over N independent batches (4 groups x 8 rollouts, rich multi c = 0.05 L_f = 1).
The gradient is recovered exactly from one SGD step with lr = 1e-6:  g = (theta_after - theta_before) / lr.
The exact gradient of J(theta) = mean over the three workflows of the exact policy value is computed by central finite
differences on the same entries. Reported per estimator and entry: mean, sd, SNR = |mean| / sd, sign agreement of the
mean with the exact gradient, and the cosine between the mean gradient vector and the exact gradient.

    python scripts/lab/grad_snr.py --n 300 --out results/lab/grad_snr.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.ad2.tabular import envs as E  # noqa: E402
from actionproof.ad2.tabular.env import policy_value  # noqa: E402
from actionproof.ad2.tabular.policy import Policy  # noqa: E402

ESTS = ["grpo", "nostd", "rloo", "igc_v1", "fork_act", "fork_act_nocrn", "cdvo_fixed"]  # learned-critic arms have an empty critic at a fixed theta: excluded
LR = 1e-6
TYPES = {0: "order", 1: "receive", 2: "pay", 3: "post"}


def thetas() -> dict[str, dict]:
    """warm: the registered warm start (rho0 = 0.3). mid: rho = 0.5 on every op type (an undecided policy)."""
    w = Policy(4, "tab", 0.3).theta
    m = Policy(4, "tab", 0.5).theta
    return {"warm": {k: v.tolist() for k, v in w.items()}, "mid": {k: v.tolist() for k, v in m.items()}}


def exact_grad(theta: dict, tasks) -> np.ndarray:
    def J(th):
        pol = Policy(4, "tab", 0.3)
        pol.theta = {k: np.array(v, dtype=float) for k, v in th.items()}
        return np.mean([policy_value(t, pol.probs) for t in tasks])
    g = np.zeros(4)
    h = 1e-5
    for tid in range(4):
        tp = {k: np.array(v, dtype=float) for k, v in theta.items()}
        tm = {k: np.array(v, dtype=float) for k, v in theta.items()}
        tp["V1"][1, 1 + tid] += h
        tm["V1"][1, 1 + tid] -= h
        g[tid] = (J(tp) - J(tm)) / (2 * h)
    return g


def one(args) -> tuple[str, str, int, list[float]]:
    from actionproof.lab.cdvo import train
    est, tname, theta, seed = args
    tasks = E.rich("multi", 0.05, 1.0)
    r = train({"tasks": tasks, "eval_tasks": tasks[:1], "est": est, "seed": 50_000 + seed, "steps": 1, "opt": "sgd",
               "lr": LR, "theta0": theta, "n_mc": 1, "fork_k": 1, "branch_frac": 0.25, "ppo_epochs": 1})
    after = np.array(r["theta"]["V1"])[1, 1:5]
    before = np.array(theta["V1"])[1, 1:5]
    return est, tname, seed, ((after - before) / LR).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="results/lab/grad_snr.json")
    a = ap.parse_args()
    t0 = time.time()
    tasks = E.rich("multi", 0.05, 1.0)
    TH = thetas()
    exact = {k: exact_grad(v, tasks).tolist() for k, v in TH.items()}
    jobs = [(e, k, TH[k], s) for e in ESTS for k in TH for s in range(a.n)]
    G: dict = {}
    with Pool(a.workers) as pool:
        for est, k, s, g in pool.imap_unordered(one, jobs, chunksize=8):
            G.setdefault((est, k), []).append(g)
    out = {"exact_grad": exact, "types": TYPES, "n": a.n, "per": {}}
    for (est, k), gs in sorted(G.items()):
        X = np.array(gs)
        mu, sd = X.mean(0), X.std(0, ddof=1)
        ex = np.array(exact[k])
        out["per"][f"{est}|{k}"] = {
            "mean": mu.tolist(), "sd": sd.tolist(), "snr": (np.abs(mu) / (sd + 1e-12)).tolist(),
            "sign_agree": [bool(np.sign(m) == np.sign(e)) for m, e in zip(mu, ex)],
            "cosine_with_exact": float(mu @ ex / (np.linalg.norm(mu) * np.linalg.norm(ex) + 1e-12)),
            "se_mean": (sd / np.sqrt(len(X))).tolist()}
    for k in TH:
        a1, a0 = out["per"].get(f"fork_act|{k}"), out["per"].get(f"fork_act_nocrn|{k}")
        if a1 and a0:
            out.setdefault("crn_variance_ratio", {})[k] = [(s1 / s0) ** 2 if s0 > 0 else None
                                                           for s1, s0 in zip(a1["sd"], a0["sd"])]
    out["seconds"] = round(time.time() - t0, 1)
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"exact": exact, "crn": out.get("crn_variance_ratio"),
                      "cos": {k: v["cosine_with_exact"] for k, v in out["per"].items()}}, indent=1))


if __name__ == "__main__":
    main()
