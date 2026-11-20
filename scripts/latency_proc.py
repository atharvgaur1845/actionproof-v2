"""Track 3 latency on ProcureSim (exploratory measurement): batch-1 wall time per episode, one task at a time, for a
model on proc_dev (same prompt, parser and loop as scripts/eval_proc.py, greedy), and the gateway's per-call overhead
(CPU, scripted agent, Observe and Enforce vs Passthrough).
    python scripts/latency_proc.py model --model runs/rft_proc/merged --adapter runs/a12_proc_rv_s0/final --tag rv_s0
    python scripts/latency_proc.py gateway
Outputs results/s2/latency_<tag>.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def stats(xs: list[float]) -> dict:
    return {"n": len(xs), "mean": statistics.mean(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95)}


def model(a) -> dict:
    spec = importlib.util.spec_from_file_location("eval_proc", ROOT / "scripts/eval_proc.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    from vllm import LLM
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.sim.prompts import PROC_PROMPTS
    from actionproof.train.merge import merge
    llm = LLM(merge(a.adapter, a.model) if a.adapter else a.model, gpu_memory_utilization=0.85, max_model_len=8192,
              dtype="bfloat16", seed=0)
    tasks = load_tasks(ROOT / "data/bench/proc/proc_dev.jsonl")[: a.n_tasks]
    ep.run(llm, tasks[:2], PROC_PROMPTS["neutral"], 1, 0.0, 16, 256, 0)  # warm-up, not timed
    per_ep, per_turn = [], []
    for t in tasks:
        t0 = time.perf_counter()
        r = ep.run(llm, [t], PROC_PROMPTS["neutral"], 1, 0.0, 16, 256, 0)[0]
        dt = time.perf_counter() - t0
        per_ep.append(dt)
        per_turn.append(dt / max(r["turns"], 1))
    return {"tag": a.tag, "split": "proc_dev", "batch": 1, "seconds_per_episode": stats(per_ep),
            "seconds_per_turn": stats(per_turn)}


def gateway(a) -> dict:
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.contracts import Contract
    from actionproof.integrity.gateway import ProcGateway
    from actionproof.integrity.sim.agents import naive
    from actionproof.integrity.sim.runtime import make_env_proc
    tasks = load_tasks(ROOT / "data/bench/proc/proc_dev.jsonl")
    out = {}
    for mode in ("passthrough", "observe", "enforce"):
        per_call = []
        for t in tasks:
            env = make_env_proc(t, f"lat-{t.task_id}")
            target = env if mode == "passthrough" else ProcGateway(env, Contract.for_task(t), mode)
            calls = []
            orig = target.call

            def timed(name, args=None, _o=orig, _c=calls):
                t0 = time.perf_counter()
                r = _o(name, args)
                _c.append(time.perf_counter() - t0)
                return r
            target.call = timed
            naive(target, t)
            per_call += calls
        out[mode] = {k: v * 1000 for k, v in stats(per_call).items() if k != "n"} | {"n_calls": len(per_call)}
    return {"tag": "gateway", "unit": "milliseconds per tool call (CPU, in-process)", "by_mode": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--model", required=True)
    m.add_argument("--adapter", default=None)
    m.add_argument("--tag", required=True)
    m.add_argument("--n-tasks", type=int, default=40)
    sub.add_parser("gateway")
    a = ap.parse_args()
    res = model(a) if a.cmd == "model" else gateway(a)
    out = ROOT / f"results/s2/latency_{res['tag']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
