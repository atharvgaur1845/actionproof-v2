"""TH1 runner: exact mean-field predictions and sampled-group training for one shard of cells.

    python scripts/theory/th1_run.py --split heldout --shard 3 --nshards 32 --workers 4

Writes results/theory/th1_<split>/shard_<k>.jsonl, one line per (cell, estimator) with the exact prediction (where the
estimator has one) and the per-seed simulated outcomes. Trainer settings (TH1 §3): plain SGD, lr 0.5, 300 steps,
B = 16 groups per step, 10 seeds per (cell, estimator).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.theory import exact as ex  # noqa: E402
from actionproof.theory import sim  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts/theory"))
from th1_grid import build, grid  # noqa: E402

SETTINGS = {"lr": 0.5, "steps": 300, "B": 16, "seeds": 10, "opt": "sgd", "kl": 0.02}
EXACT = ("grpo", "nostd", "rloo", "awrb", "gdpo")
ESTS = ("grpo", "nostd", "rloo", "batchstd", "reinforce", "ppo", "dapo", "grpo_kl", "awrb", "gdpo")


def run_cell(cell: dict) -> list[dict]:
    pb = build(cell)
    G, law, pi0 = cell["G"], cell["law"], cell["pi0"]
    J0, J1 = pb.J(0.0), pb.J(1.0)
    info = {"delta_q": pb.delta_q(), "vote": pb.vote(), "deterministic": pb.deterministic_within_state(),
            "reads_neutralise": pb.reads_neutralise(), "J0": J0, "J1": J1,
            "min_abs_beta": min(abs(pb.beta(z)) for z in range(pb.K)), "K": pb.K}
    rows = []
    for est in ESTS:
        t = time.time()
        row = {"cell": cell["id"], "est": est, **info}
        if est in EXACT:
            b = ex.bernstein(pb, G, law, est)
            traj = ex.sgd_map(b, pi0, SETTINGS["steps"], SETTINGS["lr"], G)
            row["pred"] = {"pi_T": float(traj[-1]), "endpoint": ex.endpoint(b, pi0), "U_pi0": float(ex.U_from_b(b, pi0)),
                           "dU0": float(G * b[1]), "dU1": float(-G * b[G - 1]), "sign_changes": ex.sign_changes(b),
                           "curve": [float(x) for x in traj[::10]]}
        seeds = []
        for sd in range(SETTINGS["seeds"]):
            r = sim.train(pb, G, law, est, steps=SETTINGS["steps"], B=SETTINGS["B"], lr=SETTINGS["lr"], pi0=pi0,
                          seed=1000 * sd + 7, opt=SETTINGS["opt"], kl=SETTINGS["kl"])
            seeds.append(r)
        row["sim"] = seeds
        row["seconds"] = round(time.time() - t, 2)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cells = grid(a.split)[a.shard::a.nshards]
    outd = ROOT / f"results/theory/th1_{a.split}"
    outd.mkdir(parents=True, exist_ok=True)
    out = outd / f"shard_{a.shard:03d}.jsonl"
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool, out.open("w") as f:
        for rows in pool.map(run_cell, cells):
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    print(f"shard {a.shard}: {len(cells)} cells in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
