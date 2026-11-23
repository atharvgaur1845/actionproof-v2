"""Named testbed environments (registered in docs/research/algorithm_discovery_v2/V2TB_PROTOCOL.md)."""
from __future__ import annotations

import itertools

import numpy as np

from actionproof.ad2.tabular.env import Config, OpType


def core(p: float, c: float, s: float, Lf: float = 0.0) -> Config:
    """One risky write (type 0) followed by a never-faulted journal-like write (type 1)."""
    return Config([OpType("pay", p, s, tid=0), OpType("post", 0.0, 1.0, tid=1)], c=c, Lf=Lf, mode="single",
                  tag=f"core_p{p}_c{c}_s{s}_L{Lf}")


# rich: four op types (order with visible timeouts and duplicate risk; receive; pay with false alarms; safe post)
ORDER = OpType("order", 0.08, 0.9, q_to=0.15, a=0.5, tid=0)
RECEIVE = OpType("receive", 0.15, 0.8, tid=1)
PAY = OpType("pay", 0.25, 0.5, f=0.05, tid=2)
POST = OpType("post", 0.0, 1.0, tid=3)
WORKFLOWS = {"opp": [ORDER, PAY, POST], "rpp": [RECEIVE, PAY, POST], "orp": [ORDER, RECEIVE, POST]}


def rich(mode: str, c: float, Lf: float, D: float = 0.5) -> list[Config]:
    return [Config(list(ops), c=c, Lf=Lf, D=D, mode=mode, tag=f"rich_{w}_{mode}_c{c}_L{Lf}")
            for w, ops in WORKFLOWS.items()]


def gen_tasks(split: str, seed: int = 0) -> list[Config]:
    """Generalisation tasks for the feature policy. Op types carry descriptors (p, s); the policy sees only them.
    train: p in {0.05, 0.2, 0.6}, s in {0.5, 1.0}, c in {0.01, 0.05, 0.2}, workflows of 2 risky ops + safe op.
    test_iid: same grids, other draws. test_shift: unseen p {0.1, 0.4, 0.8}, s {0.3, 0.6}, c {0.025, 0.1}.
    test_long: unseen workflow length (3 risky ops + safe op) on the training grids."""
    rng = np.random.default_rng({"train": 1, "test_iid": 2, "test_shift": 3, "test_long": 4}[split] + 100 * seed)
    if split == "test_shift":
        P, S, C = (0.1, 0.4, 0.8), (0.3, 0.6), (0.025, 0.1)
    else:
        P, S, C = (0.05, 0.2, 0.6), (0.5, 1.0), (0.01, 0.05, 0.2)
    k = 3 if split == "test_long" else 2
    n = {"train": 24, "test_iid": 12, "test_shift": 12, "test_long": 8}[split]
    out = []
    for i in range(n):
        ops = []
        for m in range(k):
            ops.append(OpType(f"t{m}", float(rng.choice(P)), float(rng.choice(S)), tid=0))
        ops.append(OpType("safe", 0.0, 1.0, tid=0))
        c = float(rng.choice(C))
        out.append(Config(ops, c=c, mode="multi", tag=f"gen_{split}_{i}"))
    return out


TB1_P = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80)
TB1_C = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)
TB1_S = (0.3, 0.6, 1.0)
TB1_L = (0.0, 1.0)


def tb1_grid():
    return list(itertools.product(TB1_P, TB1_C, TB1_S, TB1_L))
