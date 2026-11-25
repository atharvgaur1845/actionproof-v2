"""P4 evaluation (PREREGISTRATION_P4.md): P1's episode loop (`scripts/p1_eval.py` `run`/`summarise`, imported
unchanged) on the renamed-tool split `s4_api` (ApiShiftEnv, the held-out S4 surface), the unseen four-step workflow
(`s4_wf4`, `p1_test_wf4`) and the five P1 test splits. Results: `results/p4/<tag>__<split>.jsonl`; each runs once.

    python scripts/p4_eval.py --model runs/p4_dwaug_s0/merged --tag Dwaug_s0 --split s4_api --samples 4
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
SPLITS = ("s4_api", "s4_wf4", "p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    out = ROOT / f"results/p4/{a.tag}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; each model and split runs once")
    import actionproof.stage3.env as S3ENV
    from actionproof.bench.generator_proc import DATA, load_tasks
    from vllm import LLM
    if a.split == "s4_api":
        from actionproof.stage4.shift import make_env_api
        S3ENV.make_env_s3 = make_env_api  # p1_eval.run imports make_env_s3 from this module at call time
    tasks = load_tasks(DATA / f"{a.split}.jsonl")
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True)
    t0 = time.time()
    rows = P1.run(llm, tasks, a.samples, False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "phase": "p4", "model": a.model, "samples": a.samples,
         "seconds": round(time.time() - t0, 1), **P1.summarise(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "greedy_vsc": s["greedy"]["vsc"], "sampled_vsc": s["sampled"]["vsc"]}))


if __name__ == "__main__":
    main()
