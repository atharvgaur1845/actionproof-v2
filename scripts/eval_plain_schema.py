"""Evaluate A4 tasks in the TRAINING schema (plain `EcomEnv`, training system prompt), not `EcomEnvA4` (D-EVAL-1).

GRPO trains on A4 pools in the plain env (`train/grpo.py --task-jsonl`), but `run_eval_a4` serves every A4 task in
`EcomEnvA4`, whose system prompt lists `apply_store_credit` and whose ledger shows `credits`. This wrapper runs the
UNCHANGED `run_eval_a4.main` with only the env factory swapped back to the training one, so decoding, parsing, turn
loop, verifier (`verify_a4`) and seeds are identical. Credit-workflow tasks are dropped (the plain env has no credit
tool). Exploratory (task 416, SPEC §27 2026-11-13); hashed modules are not modified.

    python scripts/eval_plain_schema.py --model runs/rft_tiny_a6/merged --adapter runs/X/final \
        --tasks data/bench/a4/e_unrecoverable.jsonl --max-turns 18 --out results/eval/a7/plain/X__e_unrecoverable.jsonl
"""
from __future__ import annotations

import actionproof.bench.generator_a4 as G
import actionproof.eval.run_eval_a4 as A
from actionproof.rollout.agenttune_rollout import make_env

_load = G.load_tasks


def _plain_env(task, rollout_id, seed=0, middleware=frozenset()):
    assert not middleware, "plain-schema eval has no middleware"
    return make_env(task, rollout_id, seed=seed)


def _no_credit(path):
    return [t for t in _load(path) if (t.meta.get("a4") or {}).get("workflow") != "credit"]


A.make_env_a4 = _plain_env
G.load_tasks = _no_credit

if __name__ == "__main__":
    A.main()
