"""E1 (PREREGISTRATION_E1E2): sampled episodes on the fresh partial-payment splits.

Same environment, prompt, parser, tool-result format, 16 turns and 256 tokens as `scripts/s3_eval.py` (imported for
records and summaries, not modified). Each task is run K times at temperature 1 with a reproducible per-request seed
(seed = crc32(task_id, sample, turn)), plus once greedy (sample index -1). Every (task, sample) gets its own
environment, built from the task exactly as s3_eval builds it.
    python scripts/e1_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag s3_rv_mix_s0 \
        --split e1_pp_full --samples 8
Output: results/e1/<tag>__<split>.jsonl (one record per episode, `sample` field) + .summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
import zlib
from pathlib import Path

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.stage3.env import make_env_s3
from actionproof.stage3.rollout import start_conversation

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s3_eval", ROOT / "scripts/s3_eval.py")
S3E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3E)


def seed_of(task_id: str, sample: int, turn: int) -> int:
    return zlib.crc32(f"{task_id}|{sample}|{turn}".encode()) % 2**31


def run(llm, tasks, samples: int, max_turns: int = 16, max_tokens: int = 256) -> list[dict]:
    from vllm import SamplingParams
    from actionproof.rollout.episode import _agenttune
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        for j in [-1] + list(range(samples)):
            env = make_env_s3(t, f"s3-{t.task_id}")
            eps.append({"task": t, "sample": j, "env": env, "conv": start_conversation(env, t, PROC_NEUTRAL),
                        "turns": 0, "invalid": 0, "ended_by": "max_turns", "done": False, "n0": len(env.trace)})
    tok = llm.get_tokenizer()
    ctx = llm.llm_engine.model_config.max_model_len
    for turn in range(max_turns):
        for e in eps:
            if not e["done"] and len(tok.encode(tok.apply_chat_template(e["conv"], add_generation_prompt=True,
                                                                        tokenize=False), add_special_tokens=False)) + max_tokens > ctx:
                e["ended_by"], e["done"] = "context", True
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        params = [SamplingParams(temperature=0, max_tokens=max_tokens) if e["sample"] < 0 else
                  SamplingParams(temperature=1.0, max_tokens=max_tokens, seed=seed_of(e["task"].task_id, e["sample"], turn))
                  for e in live]
        outs = llm.chat([e["conv"] for e in live], params, use_tqdm=False)
        for e, o in zip(live, outs):
            text = o.outputs[0].text
            e["turns"] += 1
            e["conv"].append({"role": "assistant", "content": text})
            calls = parse(text) or []
            if not calls:
                e["ended_by"], e["done"] = "no_call", True
                continue
            res = []
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = None
                obs = e["env"].call(name, args) if isinstance(args, dict) else {"status": "error", "error": "invalid_arguments"}
                e["invalid"] += obs.get("error") in ("unknown_tool", "invalid_arguments")
                res.append(TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs)))
            e["conv"].append({"role": "user", "content": "\n\n".join(res)})
            if e["env"].done:
                e["ended_by"], e["done"] = "report", True
    out = []
    for e in eps:
        r = S3E.episode_record_proc(e["task"], e["env"])
        status = (r["report"] or {}).get("status")
        agent = e["env"].trace[e["n0"]:]
        r.update(sample=e["sample"], turns=e["turns"], invalid=e["invalid"], ended_by=e["ended_by"],
                 conversation=e["conv"], start_type=e["task"].meta.get("start_type"),
                 safe=S3E.R.safe_reward(None, [0], ap_episodes=[r])[0], violation=S3E.R.violation(r),
                 honest_fail=status in ("failed", "escalated") and not r["score"]["goal_met"],
                 escalated=status == "escalated", policy_calls=len(agent),
                 policy_reads=sum(x["tool"].startswith(("list_", "find_", "get_")) for x in agent))
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=("e1_pp_full", "e1_pp_rec"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--out-dir", default="results/e1")
    a = ap.parse_args()
    from vllm import LLM
    llm = LLM(a.model, gpu_memory_utilization=0.85, max_model_len=8192, dtype="bfloat16", seed=0)
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    t0 = time.time()
    recs = run(llm, tasks, a.samples)
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{a.tag}__{a.split}.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "tag": a.tag, "adapter": a.adapter}, default=str) + "\n")
    greedy = [r for r in recs if r["sample"] < 0]
    sampled = [r for r in recs if r["sample"] >= 0]
    s = {"greedy": S3E.summarize(greedy), "sampled": S3E.summarize(sampled), "tag": a.tag, "split": a.split,
         "samples": a.samples, "seconds": round(time.time() - t0)}
    (out / f"{a.tag}__{a.split}.summary.json").write_text(json.dumps(s, indent=2))
    print(a.tag, a.split, "greedy vsc", round(s["greedy"]["vsc"], 3), "sampled vsc", round(s["sampled"]["vsc"], 3))


if __name__ == "__main__":
    main()
