"""Costed-verification (CW) task pools. Base tasks (workflow, state, prompt) come from the ProcureSim generator;
the fault plan is then redrawn from the registered CW world prior (actionproof/ad/worlds.py) with its own RNG
stream, so prompts never depend on the fault.

  cw_train  400 tasks, seed 1201   training
  cw_dev    150 tasks, seed 1202   development (gates and diagnosis)
  cw_conf   300 tasks, seed 1203   confirmatory: written and hashed now, evaluated once per frozen model

    python -m actionproof.ad.pools
"""
from __future__ import annotations

import random

from actionproof.ad.worlds import sample_world, world_class
from actionproof.bench.generator_proc import make_task
from actionproof.stage3.tasks import write

POOLS = {"cw_train": (400, 1201), "cw_dev": (150, 1202), "cw_conf": (300, 1203)}


def make_pool(name: str, n: int, seed: int):
    rng, frng = random.Random(seed), random.Random(seed * 7919 + 1)
    out = []
    for i in range(n):
        t = make_task(name, i, rng)
        faults = sample_world(frng, t.meta["requested"])
        t.meta.update(faults=faults, never_commit=None, fault_class=world_class(faults), cw_prior="cw1")
        out.append(t)
    return out


def main() -> None:
    for name, (n, seed) in POOLS.items():
        write(name, make_pool(name, n, seed))


if __name__ == "__main__":
    main()
