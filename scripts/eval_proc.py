"""Batched vLLM evaluation on ProcureSim (Amendment 12). Same lockstep loop, parser and prompt format as
`eval/run_eval.py` and the GRPO rollouts; the environment and the record are the ProcureSim ones.

    python scripts/eval_proc.py --model CohereLabs/tiny-aya-global --adapter runs/a12_proc_rv_s0/final \
        --split proc_dev --out results/eval/a12/proc_rv_s0__neutral__proc_dev.jsonl
Greedy by default; `--samples 4 --temperature 1` gives the sampled gate. Confirmatory splits are refused unless
`--confirmatory` is passed, and each (model, split) may be run once (an access log in results/a12/conf_access.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from actionproof.bench.generator_proc import load_tasks
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.prompts import PROC_PROMPTS
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc
from actionproof.rollout.episode import _agenttune, system_prompt

CONFIRMATORY = {"proc_conf", "proc_combo", "proc_stress"}
ROOT = Path(__file__).resolve().parents[1]


def sample_seed(seed: int, task_id: str, sample: int, turn: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}/{task_id}/{sample}/{turn}".encode()).digest()[:4], "big")


def run(llm, tasks, base_prompt, samples, temperature, max_turns, max_tokens, seed=0):
    from vllm import SamplingParams
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        for s in range(samples):
            env = make_env_proc(t, f"ev-{t.task_id}-{s}")
            eps.append({"task": t, "env": env, "sample": s, "turns": 0, "invalid": 0, "ended_by": "max_turns",
                        "done": False, "t0": None,
                        "conv": [{"role": "system", "content": system_prompt(env, base_prompt)},
                                 {"role": "user", "content": t.prompt}]})
    tok = llm.get_tokenizer()
    ctx = llm.llm_engine.model_config.max_model_len
    for turn in range(max_turns):
        for e in eps:
            if not e["done"] and len(tok.encode(tok.apply_chat_template(
                    e["conv"], add_generation_prompt=True, tokenize=False), add_special_tokens=False)) + max_tokens > ctx:
                e["ended_by"], e["done"] = "context", True
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        params = SamplingParams(temperature=0, max_tokens=max_tokens) if temperature == 0 else [
            SamplingParams(temperature=temperature, max_tokens=max_tokens,
                           seed=sample_seed(seed, e["task"].task_id, e["sample"], turn)) for e in live]
        outs = llm.chat([e["conv"] for e in live], params, use_tqdm=False)
        for e, o in zip(live, outs):
            text = o.outputs[0].text
            e["turns"] += 1
            e["conv"].append({"role": "assistant", "content": text})
            calls = parse(text) or []
            if not calls:
                e["ended_by"], e["done"] = "no_call", True
                continue
            results = []
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = None
                obs = e["env"].call(name, args) if isinstance(args, dict) else \
                    {"status": "error", "error": "invalid_arguments"}
                e["invalid"] += obs.get("error") in ("unknown_tool", "invalid_arguments")
                results.append(TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs)))
            e["conv"].append({"role": "user", "content": "\n\n".join(results)})
            if e["env"].done:
                e["ended_by"], e["done"] = "report", True
    recs = []
    for e in eps:
        r = episode_record_proc(e["task"], e["env"])
        r.update(sample=e["sample"], turns=e["turns"], invalid=e["invalid"], ended_by=e["ended_by"],
                 conversation=e["conv"], workflow=e["task"].workflow, split=e["task"].split)
        recs.append(r)
    return recs


def ratio(recs, key):
    n = sum(r["score"][key][0] for r in recs)
    d = sum(r["score"][key][1] for r in recs)
    return {"num": n, "den": d, "rate": n / d if d else None}


def summarize(recs) -> dict:
    n = len(recs)
    calls = sum(r["n_model_calls"] for r in recs)

    def block(rs):
        m = len(rs)
        return {"n": m, "vsc": sum(r["score"]["vsc"] for r in rs) / m, "goal_met": sum(r["score"]["goal_met"] for r in rs) / m,
                "r_v": sum(r["score"]["r_v"] for r in rs) / m, "r_a": sum(r["score"]["r_a"] for r in rs) / m,
                "false_success": sum(r["score"]["false_success"] for r in rs) / m,
                "duplicate": sum(r["score"]["duplicate_irreversible"] for r in rs) / m,
                "collateral": sum(r["score"]["collateral"] for r in rs) / m,
                "phantom_readback": ratio(rs, "phantom_readback"), "benign_readback": ratio(rs, "benign_readback"),
                "lost_ack_readback": ratio(rs, "lost_ack_readback"), "precommit_recovery": ratio(rs, "precommit_recovery")}
    by = defaultdict(list)
    for r in recs:
        by[r["fault_class"]].append(r)
    return {**block(recs), "report_rate": sum(r["ended_by"] == "report" for r in recs) / n,
            "tool_call_validity": 1 - sum(r["invalid"] for r in recs) / max(calls, 1),
            "mean_calls": calls / n, "mean_turns": sum(r["turns"] for r in recs) / n,
            "by_fault_class": {k: block(v) for k, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--prompt", default="neutral", choices=list(PROC_PROMPTS))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.split in CONFIRMATORY:
        if not a.confirmatory:
            raise SystemExit(f"{a.split} is confirmatory: pass --confirmatory (one run per frozen model)")
        log = ROOT / "results" / "a12" / "conf_access.jsonl"
        key = {"model": a.model, "adapter": a.adapter, "split": a.split, "prompt": a.prompt}
        prior = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        if any({k: p[k] for k in key} == key for p in prior):
            raise SystemExit(f"confirmatory split already run for {key}; re-runs are forbidden (resume from the file)")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(json.dumps({**key, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")

    from vllm import LLM
    from actionproof.train.merge import merge
    weights = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(weights, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, dtype="bfloat16", seed=a.seed)
    tasks = load_tasks(ROOT / "data" / "bench" / "proc" / f"{a.split}.jsonl")[: a.n_tasks]
    t0 = time.time()
    recs = run(llm, tasks, PROC_PROMPTS[a.prompt], a.samples, a.temperature, a.max_turns, a.max_tokens, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "model": a.model, "adapter": a.adapter, "prompt": a.prompt}) + "\n")
    s = {**summarize(recs), "model": a.model, "adapter": a.adapter, "prompt": a.prompt, "split": a.split,
         "samples": a.samples, "temperature": a.temperature, "seconds": round(time.time() - t0)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: s[k] for k in ("n", "vsc", "r_v", "r_a", "phantom_readback", "false_success", "report_rate",
                                         "tool_call_validity", "seconds")}))


if __name__ == "__main__":
    main()
