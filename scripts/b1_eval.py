"""B1 (PREREGISTRATION_B1.md): prompted, untrained models on the frozen P1 test, with P1's episode loop unchanged.

Imports `scripts/p1_eval.py` (frozen by P1) and reuses its `run` and `summarise` exactly: same system prompt
(PROC_NEUTRAL), tool format, 16 turns, 256 new tokens, one greedy plus `--samples` sampled runs at T = 1, and the
deterministic verifier. Only the vLLM engine settings differ (memory share, optional fp8 weights for 32B on one
80 GB GPU), and results go to `results/b1/<tag>__<split>.jsonl` so no P1 file is touched. Each model and split runs once.

    python scripts/b1_eval.py --model CohereLabs/aya-expanse-32b --tag aya32b --split p1_test_full --samples 4 --fp8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p1_eval", ROOT / "scripts/p1_eval.py")
P1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    if not a.split.startswith("p1_test_"):
        raise SystemExit("B1 runs only the frozen P1 test splits")
    out = ROOT / f"results/b1/{a.tag}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; each model and split runs once")
    from actionproof.bench.generator_proc import DATA, load_tasks
    from vllm import LLM
    tasks = load_tasks(DATA / f"{a.split}.jsonl")
    kw = {"quantization": "fp8"} if a.fp8 else {}
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=True, **kw)
    t0 = time.time()
    rows = P1.run(llm, tasks, a.samples, False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "phase": "b1", "model": a.model, "fp8": a.fp8, "samples": a.samples,
         "seconds": round(time.time() - t0, 1), **P1.summarise(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "greedy_vsc": s["greedy"]["vsc"], "sampled_vsc": s["sampled"]["vsc"]}))


if __name__ == "__main__":
    main()
