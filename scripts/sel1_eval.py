"""SEL1 model evaluation (PREREGISTRATION_SEL1.md). TXN v1's protocol unchanged: `scripts/txn_eval.py`'s episode loop
(p1_eval's `run`: prompts, tool schemas, parser, tool-result format, 16 turns, 256 new tokens, 1 greedy + `--samples`
sampled episodes at T = 1 with p1_eval's per-(task, sample, turn) seeds) and its scoring (`truth.txn_score`,
invariants, decision record).

Only the env factory and the output directory differ:
- `--split txn_test`: the locked TXN test, env `make_env_txn` (exactly as scripts/txn_eval.py);
- `--split sel1_test`: the sealed SEL1 test, env `actionproof.sel1.env.make_env_sel` (adds the transient outage).

Output: results/sel1/<tag>__<split>.jsonl and .summary.json. Each tag and split is run once (never overwritten).

    python scripts/sel1_eval.py --model runs/p4_dwaug_s0/merged --adapter runs/sel1_SEL_s0/final --tag SEL_s0 --split sel1_test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("txn_eval", ROOT / "scripts/txn_eval.py")
TX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TX)
P1 = TX.P1
SPLITS = ("txn_test", "sel1_test")


def run_sel(llm, tasks, samples: int, split: str) -> list[dict]:
    import actionproof.stage3.env as S3
    from actionproof.sel1.env import make_env_sel
    from actionproof.txn.env import make_env_txn
    orig = S3.make_env_s3
    if split == "sel1_test":
        S3.make_env_s3 = lambda task, rid, seed=None: make_env_sel(task, rid, seed=seed)
    else:
        S3.make_env_s3 = lambda task, rid, seed=None: make_env_txn(task, rid, seed=seed)
    try:
        rows = P1.run(llm, tasks, samples, recon=False)
    finally:
        S3.make_env_s3 = orig
    return [TX.annotate(r) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    out = ROOT / f"results/sel1/{a.tag}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; each tag and split is run once")
    from actionproof.bench.generator_proc import DATA, load_tasks
    tasks = load_tasks(DATA / f"{a.split}.jsonl")
    t0 = time.time()
    from vllm import LLM
    from actionproof.train.merge import merge
    model = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=True)
    rows = run_sel(llm, tasks, a.samples, a.split)
    for r in rows:
        r["tag"] = a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "model": a.model, "adapter": a.adapter, "samples": a.samples,
         "n_tasks": len(tasks), "seconds": round(time.time() - t0, 1), **P1.summarise(rows), "txn": TX.txn_summary(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "STR": s["txn"]["all"]["STR"], "CER": s["txn"]["all"]["CER"]}))


if __name__ == "__main__":
    main()
