"""Amendment 4 evaluation runner. Reuses the hash-registered `run_eval.run` / `summarize` UNCHANGED: only the
module-level env factory and episode-record functions it calls are swapped for the A4 versions (held-out tool
env, A4 verifier, optional middleware), so decoding, parsing, turn loop and seeds are byte-identical to v2.

    python -m actionproof.eval.run_eval_a4 --model runs/rft_tiny_v2/merged --adapter runs/X/final \
        --tasks data/bench/a4/e_heldout.jsonl --max-turns 16 --middleware raw --out results/eval/a4/X__raw__e_heldout.jsonl
    # registered dev split (standard env) with middleware:
    python -m actionproof.eval.run_eval_a4 ... --split test_iid --max-turns 10 --middleware vbr --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import actionproof.eval.run_eval as RE
from actionproof.eval.a4_metrics import summarize_a4
from actionproof.eval.a4_runtime import episode_record_a4, make_env_a4


def file_sha(p: str | None) -> str | None:
    if not p or not Path(p).exists():
        return None
    h = hashlib.sha256()
    for f in sorted(Path(p).rglob("*")) if Path(p).is_dir() else [Path(p)]:
        if f.is_file() and f.suffix in (".safetensors", ".jsonl", ".json"):
            h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tasks", help="A4 task jsonl (data/bench/a4/*.jsonl)")
    src.add_argument("--split", help="registered split name (make_split)")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--prompt", default="neutral", choices=list(RE.PROMPTS))
    ap.add_argument("--middleware", default="none", choices=("none", "raw", "vbr", "raw+vbr"))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.tasks:
        from actionproof.bench.generator_a4 import load_tasks
        tasks = load_tasks(a.tasks)[: a.n_tasks]
    else:
        from actionproof.bench.generator import make_split
        tasks = make_split(a.split, a.n_tasks)
    mw = frozenset() if a.middleware == "none" else frozenset(a.middleware.split("+"))
    RE.make_env = lambda t, rid, seed=0: make_env_a4(t, rid, seed=seed, middleware=mw)  # swap factory only
    RE.episode_record = episode_record_a4

    from vllm import LLM
    from actionproof.train.merge import merge
    weights = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(weights, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len, dtype="bfloat16", seed=a.seed)
    t0 = time.time()
    recs = RE.run(llm, tasks, RE.PROMPTS[a.prompt], a.samples, a.temperature, a.max_turns, a.max_tokens, None, a.seed)
    secs = time.time() - t0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "model": a.model, "adapter": a.adapter, "prompt": a.prompt,
                                "middleware": a.middleware}) + "\n")
    s = {**RE.summarize(recs), "a4": summarize_a4(recs), "model": a.model, "adapter": a.adapter,
         "adapter_sha256": file_sha(a.adapter), "tasks": a.tasks, "tasks_sha256": file_sha(a.tasks), "split": a.split,
         "middleware": a.middleware, "samples": a.samples, "temperature": a.temperature, "max_turns": a.max_turns,
         "seed": a.seed, "seconds": round(secs), "gen_chars": sum(len(m["content"]) for r in recs
                                                              for m in r["conversation"] if m["role"] == "assistant")}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: s[k] for k in ("n", "vsc", "fcr", "dsr", "ivr", "verify_calls", "middleware")}))


if __name__ == "__main__":
    main()
