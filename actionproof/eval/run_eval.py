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
import hashlib
import json
import time
from pathlib import Path

from actionproof.bench.generator import make_split
from actionproof.bench.prompts import NEUTRAL, TOOL_RESULT_FORMAT, VERIFY_PROMPTS
from actionproof.rollout.agenttune_rollout import episode_record, make_env
from actionproof.env.tools_ecom import TOOL_META
from actionproof.interp.guard import NUDGE
from actionproof.rollout.episode import _agenttune, system_prompt

PROMPTS = {"neutral": NEUTRAL, **VERIFY_PROMPTS}


def sample_seed(seed: int, task_id: str, sample: int, turn: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}/{task_id}/{sample}/{turn}".encode()).digest()[:4], "big")


def run(llm, tasks, base_prompt: str, samples: int, temperature: float, max_turns: int,
        max_tokens: int, lora=None, seed: int = 0, guard=None):
    from vllm import SamplingParams
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        for s in range(samples):
            env = make_env(t, f"ev-{t.task_id}-{s}", seed=seed)  # env randomness depends on the task only, so samples are comparable
            conv = [{"role": "system", "content": system_prompt(env, base_prompt)},
                    {"role": "user", "content": t.prompt}]
            eps.append({"task": t, "env": env, "conv": conv, "sample": s, "turns": 0,
                        "unparsed": 0, "invalid": 0, "ended_by": "max_turns", "done": False, "nudges": 0})
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)

    def params(live, turn):
        """Greedy: one shared SamplingParams (unchanged). Sampling: an independent, reproducible seed per
        (task, sample, turn). A single shared seed made every request draw the same random stream, so the
        k samples of a task were copies and draws were correlated across tasks (fixed 2026-11-05)."""
        if temperature == 0:
            return sp
        return [SamplingParams(temperature=temperature, max_tokens=max_tokens,
                               seed=sample_seed(seed, e["task"].task_id, e["sample"], turn))
                for e in live]
    tok = llm.get_tokenizer()
    ctx = llm.llm_engine.model_config.max_model_len
    for _turn in range(max_turns):
        for e in eps:  # same guard as GRPO rollouts: an episode that would overflow the context ends
            if not e["done"] and len(tok.encode(tok.apply_chat_template(
                    e["conv"], add_generation_prompt=True, tokenize=False), add_special_tokens=False)) + max_tokens > ctx:
                e["ended_by"], e["done"] = "context", True
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        outs = llm.chat([e["conv"] for e in live], params(live, _turn), use_tqdm=False,
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
            results, block_obs = [], []
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
                block_obs.append({**obs, "_write": TOOL_META.get(name, (None, ""))[1] == "write"})
            e["conv"].append({"role": "user", "content": "\n\n".join(results)})
            if guard is not None and not e["env"].done and guard(e["conv"], block_obs):
                e["conv"][-1]["content"] += NUDGE
                e["nudges"] += 1
            if e["env"].done:
                e["ended_by"], e["done"] = "report", True
    records = []
    for e in eps:
        r = episode_record(e["task"], e["env"])
        r.update(sample=e["sample"], turns=e["turns"], nudges=e["nudges"], unparsed=e["unparsed"], invalid=e["invalid"],
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
    ap.add_argument("--guard-probe", default=None, help="probe npz: enable the probe-triggered verification guard")
    ap.add_argument("--guard-tau", type=float, default=None)
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
    guard = None
    if a.guard_probe:
        from actionproof.interp.guard import ProbeGuard
        guard = ProbeGuard(weights, a.guard_probe, a.guard_tau)
    recs = run(llm, tasks, PROMPTS[a.prompt], a.samples, a.temperature, a.max_turns, a.max_tokens, lora, a.seed, guard)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "model": a.model, "adapter": a.adapter, "prompt": a.prompt}) + "\n")
    s = {**summarize(recs), "nudges_per_ep": sum(r["nudges"] for r in recs) / len(recs),
         "guard": {"probe": a.guard_probe, "tau": guard.tau, "checks": guard.checks, "fires": guard.fires} if guard else None,
         "model": a.model, "adapter": a.adapter, "prompt": a.prompt, "split": a.split,
         "samples": a.samples, "temperature": a.temperature, "seconds": round(time.time() - t0)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
