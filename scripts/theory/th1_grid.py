"""TH1 cells: random verification problems x group laws x group sizes x initial policies.

    python scripts/theory/th1_grid.py --split dev       # 60 cells, RNG seed 101 (development: tune trainer settings)
    python scripts/theory/th1_grid.py --split heldout   # 400 cells + dirmult twins, RNG seed 20261124 (registered)

The held-out grid is a pure function of its seed; it is written by the analysis job and hashed by freeze_th1.py
together with this file. Strata (fixed after the first dev run showed the exact regimes were too rare, 2/72 cells):
  broad 45%        unrestricted draws
  exact 25%        deterministic within-state returns (q0 = 0, f = 0, d = 1, s in {0, 1} or repairability split)
  neutralise 15%   reads neutralise the state (s = d = 1, f = q0 = 0, one read cost), law not shared, pi0 >= 0.3
  adversarial 15%  c near the stake, vote near zero, c at the eps boundary, or G = 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.theory.exact import categories  # noqa: E402
from actionproof.theory.problems import verification  # noqa: E402

SEEDS = {"dev": (101, 60), "heldout": (20261124, 600)}
BUDGET = 150_000  # max category compositions per exact evaluation


def ncomp(C: int, G: int) -> int:
    return math.comb(G + C - 1, C - 1)


def build(cell: dict):
    return verification(cell["p"], cell["L"], cell["s"], d=cell["d"], c=cell["c"], f=cell["f"], D=cell["D"],
                        q0=cell["q0"], split_repair=cell["split_repair"], uninformative=cell["uninformative"])


def draw(rng: np.random.Generator, i: int, split: str) -> dict:
    u0 = rng.random()
    stratum = "broad" if u0 < 0.45 else "exact" if u0 < 0.70 else "neutralise" if u0 < 0.85 else "adversarial"
    adv = stratum == "adversarial"
    nf = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
    mass = float(rng.uniform(0.05, 0.9))
    p = (rng.dirichlet(np.ones(nf)) * mass).tolist()
    L, s, d = [], [], []
    for _ in range(nf):
        u = rng.random()
        L.append(0.0 if u < 0.4 else float(rng.uniform(0, 3)) if u < 0.7 else float(rng.uniform(3, 30)))
        u = rng.random()
        s.append(1.0 if u < 0.5 else 0.0 if u < 0.7 else float(rng.uniform(0.2, 0.95)))
        d.append(1.0 if rng.random() < 0.6 else float(rng.uniform(0.5, 1.0)))
    f = 0.0 if rng.random() < 0.6 else float(rng.uniform(0, 0.3))
    D = float(rng.uniform(0, 2)) if f > 0 else 0.0
    if rng.random() < 0.15:
        c = [float(math.exp(rng.uniform(math.log(0.005), math.log(0.6)))) for _ in range(nf + 1)]
    else:
        c = float(math.exp(rng.uniform(math.log(0.005), math.log(0.6))))
    q0 = 0.0 if rng.random() < 0.7 else float(rng.uniform(0, 0.3))
    split_repair = bool(rng.random() < 0.3)
    unin = bool(rng.random() < 0.05)
    G = int(rng.choice([2, 4, 8, 16], p=[0.15, 0.3, 0.4, 0.15]))
    law_kind = str(rng.choice(["shared", "iid", "mix", "dirmult"], p=[0.3, 0.2, 0.25, 0.25]))
    rho = float(rng.uniform(0.05, 0.95))
    pi0 = float(rng.choice([0.1, 0.3, 0.5, 0.7]))
    if stratum == "exact":
        q0, f, D, d = 0.0, 0.0, 0.0, [1.0] * nf
        s = [x if x in (0.0, 1.0) else x for x in s]
        if any(0 < x < 1 for x in s):
            split_repair = True
    if stratum == "neutralise":
        q0, f, D, d, s = 0.0, 0.0, 0.0, [1.0] * nf, [1.0] * nf
        c = float(math.exp(rng.uniform(math.log(0.005), math.log(0.3))))
        law_kind = str(rng.choice(["iid", "mix", "dirmult"]))
        pi0 = float(rng.choice([0.3, 0.5, 0.7]))
    if adv:
        kind = int(rng.integers(4))
        if kind == 0:   # cost near the stake of the first fault
            c = float(rng.uniform(0.8, 1.2) * (1 + L[0]))
        elif kind == 1:  # vote near zero
            p = [float(rng.uniform(0.45, 0.55))]
            L, s, d, nf = [L[0]], [1.0], [1.0], 1
            if isinstance(c, list):
                c = c[0]
        elif kind == 2:  # eps boundary
            c = float(rng.uniform(0.0003, 0.002))
        else:
            G = 2
    law = law_kind if law_kind in ("shared", "iid") else f"{law_kind}:{rho:.4f}"
    cell = {"id": f"{split}-{i:04d}", "split": split, "stratum": stratum, "p": p, "L": L, "s": s, "d": d, "f": f,
            "D": D, "c": c, "q0": q0, "split_repair": split_repair, "uninformative": unin, "G": G, "law": law,
            "rho": None if law_kind in ("shared", "iid") else rho, "pi0": pi0}
    while G > 2 and ncomp(len(categories(build(cell), "grpo")["q"]), G) > BUDGET:
        G //= 2
        cell["G"] = G
    return cell


def grid(split: str) -> list[dict]:
    seed, n = SEEDS[split]
    rng = np.random.default_rng(seed)
    cells = [draw(rng, i, split) for i in range(n)]
    twins = []
    for c in cells:  # dirmult twin of every mixture cell (same marginal and pairwise ICC): Proposition 3.5
        if c["law"].startswith("mix:"):
            t = dict(c, id=c["id"] + "-dm", law="dirmult:" + c["law"][4:], twin_of=c["id"])
            twins.append(t)
    return cells + twins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SEEDS), required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    g = grid(a.split)
    out = Path(a.out or ROOT / f"results/theory/th1_cells_{a.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g, indent=0))
    print(len(g), "cells ->", out)


if __name__ == "__main__":
    main()
