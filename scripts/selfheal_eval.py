"""Self-heal evaluation on loop-pool and gate splits (GPU, vLLM). The episode loop, prompts, parser, 16 turns and
256 new tokens are scripts/p1_eval.py's; TXN tasks run on the TXN env exactly as scripts/txn_eval.py does (its
`run_txn` and `annotate`). The env is chosen per split from the task metadata (`actionproof.selfheal.envs`).

    python scripts/selfheal_eval.py --model M [--adapter A] --tag D_s0 --split sh_pool_r1_txn --samples 4

Output: results/selfheal/eval/<tag>__<split>.jsonl (+ .summary.json). Only sh_* splits are accepted (the frozen P1
and TXN test splits are refused). A gate split (sh_gate_*) is run once per tag: an existing output is never
overwritten.
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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    if not a.split.startswith("sh_"):
        raise SystemExit("selfheal_eval runs sh_* splits only")
    out = ROOT / f"results/selfheal/eval/{a.tag}__{a.split}.jsonl"
    if a.split.startswith("sh_gate_") and (out.exists() or a.n_tasks is not None):
        raise SystemExit(f"{out}: a gate split is run once per tag and in full")
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.selfheal.envs import spec_for
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    envs = {spec_for(t).name for t in tasks}
    if len(envs) != 1:
        raise SystemExit(f"{a.split}: mixed envs {envs}")
    from actionproof.train.merge import merge
    from vllm import LLM
    model = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=True)
    P1 = _load("p1_eval")
    t0 = time.time()
    if envs == {"txn_v1"}:
        TX = _load("txn_eval")
        rows = TX.run_txn(llm, tasks, a.samples)
        extra = {"txn": TX.txn_summary(rows)}
    else:
        rows = P1.run(llm, tasks, a.samples, recon=False)
        extra = {}
    for r in rows:
        r["tag"] = a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "env": envs.pop(), "model": a.model, "adapter": a.adapter,
         "samples": a.samples, "seconds": round(time.time() - t0, 1), **P1.summarise(rows), **extra}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "greedy_vsc": s["greedy"]["vsc"],
                      "sampled_vsc": (s["sampled"] or {}).get("vsc")}))


if __name__ == "__main__":
    main()
