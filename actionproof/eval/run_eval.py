"""Batched multi-turn evaluation / teacher generation with vLLM (spec §7, §13.1, §15).

All episodes advance in lockstep: one batched `LLM.chat` call per turn over every unfinished
episode, so a 32B teacher or a 7-model leaderboard runs at vLLM throughput instead of one
conversation at a time. Prompt format, schemas and the tool-call parser are AgentTune's
(via actionproof.rollout.episode), identical to training.

    python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --split test_iid \
        --prompt neutral --samples 1 --out results/eval/tiny_global_neutral_test_iid.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from actionproof.bench.generator import make_split
from actionproof.bench.prompts import NEUTRAL, TOOL_RESULT_FORMAT, VERIFY_PROMPTS
from actionproof.rollout.agenttune_rollout import episode_record, make_env
from actionproof.rollout.episode import _agenttune, system_prompt

PROMPTS = {"neutral": NEUTRAL, **VERIFY_PROMPTS}


def run(llm, tasks, base_prompt: str, samples: int, temperature: float, max_turns: int,
        max_tokens: int, lora=None, seed: int = 0):
    from vllm import SamplingParams
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        for s in range(samples):
            env = make_env(t, f"ev-{t.task_id}-{s}", seed=seed * 1000 + s)
            conv = [{"role": "system", "content": system_prompt(env, base_prompt)},
                    {"role": "user", "content": t.prompt}]
            eps.append({"task": t, "env": env, "conv": conv, "sample": s, "turns": 0,
                        "unparsed": 0, "invalid": 0, "ended_by": "max_turns", "done": False})
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
    for _turn in range(max_turns):
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        outs = llm.chat([e["conv"] for e in live], sp, use_tqdm=False,
                        **({"lora_request": lora} if lora else {}))
        for e, o in zip(live, outs):
            text = o.outputs[0].text
            e["turns"] += 1
            e["conv"].append({"role": "assistant", "content": text})
            calls = parse(text) or []
            if not calls:
                e["unparsed"] += 1
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
    records = []
    for e in eps:
        r = episode_record(e["task"], e["env"])
        r.update(sample=e["sample"], turns=e["turns"], unparsed=e["unparsed"], invalid=e["invalid"],
                 ended_by=e["ended_by"], conversation=e["conv"], workflow=e["task"].workflow,
                 split=e["task"].split, benign=e["task"].meta["benign"])
        records.append(r)
    return records


def summarize(records) -> dict:
    n = len(records)
    fails = [r for r in records if r["verdict"]["G"] == 0]
    return {"n": n, "vsc": sum(r["vsc"] for r in records) / n,
            "fcr": sum(r["verdict"]["F_over"] for r in fails) / max(len(fails), 1),
            "dsr": sum(r["verdict"]["D"] > 0 for r in records) / n,
            "ivr": sum(r["verdict"]["I"] > 0 for r in records) / n,
            "report_rate": sum(r["ended_by"] == "report" for r in records) / n,
            "mean_turns": sum(r["turns"] for r in records) / n,
            "verify_calls": sum(r["n_verify"] for r in records) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test_iid")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--prompt", default="neutral", choices=list(PROMPTS))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM
    from actionproof.train.merge import merge
    # vLLM LoRA crashes on Cohere's tied embeddings, so adapters are served merged.
    weights = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(weights, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, dtype="bfloat16", seed=a.seed)
    lora = None
    tasks = make_split(a.split, a.n_tasks)
    t0 = time.time()
    recs = run(llm, tasks, PROMPTS[a.prompt], a.samples, a.temperature, a.max_turns, a.max_tokens, lora, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "model": a.model, "adapter": a.adapter, "prompt": a.prompt}) + "\n")
    s = {**summarize(recs), "model": a.model, "adapter": a.adapter, "prompt": a.prompt, "split": a.split,
         "samples": a.samples, "temperature": a.temperature, "seconds": round(time.time() - t0)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
