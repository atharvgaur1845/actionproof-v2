"""Amendment 7 split (PREREGISTRATION_A7 §3). Built from generator_a4's task constructor, unchanged.

  a7_credit_demo   20 fault-free store-credit tasks (the credit workflow of e_heldout), from a fresh seed stream, so
                   ids and worlds are disjoint from a4_train_A/B, a6_demo and every evaluation split. Tasks carry
                   `domain: a4`, so they run in EcomEnvA4, the only environment that has `apply_store_credit`.

    python -m actionproof.bench.generator_a7 --out data/bench/a7
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from actionproof.bench.generator import Task
from actionproof.bench.generator_a4 import _task

CREDIT_DEMO_SEED, N_CREDIT_DEMO = 41, 20
SPLITS = ("a7_credit_demo",)


def make_credit_demo_pool() -> list[Task]:
    return [_task("a7_credit_demo", i, random.Random(f"actionproof-a7/credit-demo/{CREDIT_DEMO_SEED}/{i}"), "credit",
                  False, "OK", None, None, None, None, True) for i in range(N_CREDIT_DEMO)]


def make_a7_split(split: str) -> list[Task]:
    return {"a7_credit_demo": make_credit_demo_pool}[split]()


def split_bytes(split: str) -> bytes:
    return "".join(json.dumps(t.to_json(), sort_keys=True) + "\n" for t in make_a7_split(split)).encode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bench/a7")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (out / f"{s}.jsonl").write_bytes(split_bytes(s))
        print(s, len(make_a7_split(s)))


if __name__ == "__main__":
    main()
