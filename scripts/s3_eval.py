"""Stage 3 evaluation: the lockstep vLLM loop of `scripts/eval_proc.py` (same parser, tool-result format, greedy,
16 turns, 256 tokens), on the Stage 3 environment, starting from each task's scripted `start_prefix` when it has one.
Reports, per split and per start type / fault class: VSC, the `safe` reward, false completion, violations
(duplicate, collateral, wrong amount or payee), honest failure, escalation, and calls per episode.

    python scripts/s3_eval.py --model runs/rft_proc/merged --adapter runs/s3_safe_mix_s0/final --tag safe_mix_s0 \
        --split s3_rec_dev --out-dir results/s3/dev
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.stage3 import rewards as R
from actionproof.stage3.env import make_env_s3
from actionproof.stage3.rollout import start_conversation

ROOT = Path(__file__).resolve().parents[1]


def run(llm, tasks, max_turns=16, max_tokens=256):
    from vllm import SamplingParams
    from actionproof.rollout.episode import _agenttune
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        env = make_env_s3(t, f"s3-{t.task_id}")
        eps.append({"task": t, "env": env, "conv": start_conversation(env, t, PROC_NEUTRAL), "turns": 0, "invalid": 0,
                    "ended_by": "max_turns", "done": False, "n0": len(env.trace)})
    tok = llm.get_tokenizer()
    ctx = llm.llm_engine.model_config.max_model_len
    for _ in range(max_turns):
        for e in eps:
            if not e["done"] and len(tok.encode(tok.apply_chat_template(e["conv"], add_generation_prompt=True,
                                                                        tokenize=False), add_special_tokens=False)) + max_tokens > ctx:
                e["ended_by"], e["done"] = "context", True
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        outs = llm.chat([e["conv"] for e in live], SamplingParams(temperature=0, max_tokens=max_tokens), use_tqdm=False)
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
        r = episode_record_proc(e["task"], e["env"])
        status = (r["report"] or {}).get("status")
        agent = e["env"].trace[e["n0"]:]
        r.update(turns=e["turns"], invalid=e["invalid"], ended_by=e["ended_by"], conversation=e["conv"],
                 start_type=e["task"].meta.get("start_type"), safe=R.safe_reward(None, [0], ap_episodes=[r])[0],
                 violation=R.violation(r), honest_fail=status in ("failed", "escalated") and not r["score"]["goal_met"],
                 escalated=status == "escalated", policy_calls=len(agent),
                 policy_reads=sum(x["tool"].startswith(("list_", "find_", "get_")) for x in agent))
        out.append(r)
    return out


def block(rs):
    n = len(rs)
    f = lambda k: sum(float(r[k]) for r in rs) / n
    g = lambda k: sum(float(r["score"][k]) for r in rs) / n
    return {"n": n, "vsc": g("vsc"), "safe": f("safe"), "false_success": g("false_success"), "violation": f("violation"),
            "duplicate": g("duplicate_irreversible"), "honest_fail": f("honest_fail"), "escalated": f("escalated"),
            "policy_calls": f("policy_calls"), "policy_reads": f("policy_reads"),
            "report_rate": sum(r["ended_by"] == "report" for r in rs) / n}


def summarize(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["fault_class"]].append(r)
    calls = sum(r["n_model_calls"] for r in recs)
    return {**block(recs), "tool_call_validity": 1 - sum(r["invalid"] for r in recs) / max(calls, 1),
            "by_class": {k: block(v) for k, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--out-dir", default="results/s3/dev")
    a = ap.parse_args()
    from vllm import LLM
    from actionproof.train.merge import merge
    llm = LLM(merge(a.adapter, a.model) if a.adapter else a.model, gpu_memory_utilization=0.85, max_model_len=8192,
              dtype="bfloat16", seed=0)
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    t0 = time.time()
    recs = run(llm, tasks)
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{a.tag}__{a.split}.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps({**r, "tag": a.tag, "adapter": a.adapter}, default=str) + "\n")
    s = {**summarize(recs), "tag": a.tag, "split": a.split, "seconds": round(time.time() - t0)}
    (out / f"{a.tag}__{a.split}.summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "by_class"}))


if __name__ == "__main__":
    main()
