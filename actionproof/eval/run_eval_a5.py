"""Amendment 5 evaluation runner. Like run_eval_a4, it reuses the hash-registered `run_eval.run` / `summarize`
unchanged and swaps only the module-level env factory / record functions (A5 env, optional A5 middleware).

    python -m actionproof.eval.run_eval_a5 --model runs/rft_tiny_a5/merged --adapter runs/X/final \
        --tasks data/bench/a5/a5_eval.jsonl --max-turns 18 [--middleware raw|vbr|raw+vbr] [--prompt evidence_contract] \
        --out results/eval/a5/X.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import actionproof.eval.run_eval as RE
from actionproof.bench.prompts_a5 import EVIDENCE_CONTRACT
from actionproof.eval.a5_middleware import make_env_a5_mw
from actionproof.eval.a5_runtime import episode_record_a5
from actionproof.eval.run_eval_a4 import file_sha

A5_KEYS = ("proof_valid", "goal_final", "exactly_once", "duplicate", "supported", "cert_valid", "false_completion",
           "unsupported_completion", "underclaim", "safe_abstention", "unnecessary_abstention", "external_effect",
           "evidence_calls", "waits", "sim_seconds")


def summarize_a5(recs: list[dict]) -> dict:
    def agg(rs):
        out = {k: sum(r["a5v"][k] for r in rs) / len(rs) for k in A5_KEYS}
        att = [r["a5v"]["attribution_correct"] for r in rs if r["a5v"]["attribution_correct"] is not None]
        out["attribution_correct"] = sum(att) / len(att) if att else None
        out["collateral"] = sum(bool(r["a5v"]["collateral"]) for r in rs) / len(rs)
        out["unauthorized"] = sum(any(r["a5v"]["violations"].values()) for r in rs) / len(rs)
        out["model_calls"] = sum(r["n_model_calls"] for r in rs) / len(rs)
        out["harness_calls"] = sum(r["n_harness"] for r in rs) / len(rs)
        out["n"] = len(rs)
        return out
    by = defaultdict(list)
    for r in recs:
        by[r["a5"]["scenario"]].append(r)
    return {"all": agg(recs), "by_scenario": {k: agg(v) for k, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--prompt", default="neutral", choices=[*RE.PROMPTS, "evidence_contract"])
    ap.add_argument("--middleware", default="none", choices=("none", "raw", "vbr", "raw+vbr"))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=18)
    ap.add_argument("--max-tokens", type=int, default=256)  # = GRPO rollout cap
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from actionproof.bench.generator_a4 import load_tasks
    tasks = load_tasks(a.tasks)[: a.n_tasks]
    prompts = {**RE.PROMPTS, "evidence_contract": EVIDENCE_CONTRACT}
    mw = frozenset() if a.middleware == "none" else frozenset(a.middleware.split("+"))
    RE.make_env = lambda t, rid, seed=0: make_env_a5_mw(t, rid, seed=seed, middleware=mw)
    RE.episode_record = episode_record_a5

    from vllm import LLM
    from actionproof.train.merge import merge
    weights = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(weights, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              dtype="bfloat16", seed=a.seed)
    t0 = time.time()
    recs = RE.run(llm, tasks, prompts[a.prompt], a.samples, a.temperature, a.max_turns, a.max_tokens, None, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "model": a.model, "adapter": a.adapter, "prompt": a.prompt,
                                "middleware": a.middleware}) + "\n")
    s = {"a5": summarize_a5(recs), "ended_by": dict(__import__("collections").Counter(r["ended_by"] for r in recs)),
         "model": a.model, "adapter": a.adapter, "adapter_sha256": file_sha(a.adapter), "tasks": a.tasks,
         "tasks_sha256": file_sha(a.tasks), "prompt": a.prompt, "middleware": a.middleware, "samples": a.samples,
         "temperature": a.temperature, "max_turns": a.max_turns, "seed": a.seed, "seconds": round(time.time() - t0)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    al = s["a5"]["all"]
    print(json.dumps({k: (round(al[k], 3) if isinstance(al[k], float) else al[k]) for k in
                      ("n", "proof_valid", "goal_final", "duplicate", "false_completion", "safe_abstention",
                       "unnecessary_abstention", "cert_valid", "evidence_calls")}))


if __name__ == "__main__":
    main()
