"""Evaluation on the costed-verification testbed: the lockstep vLLM loop of scripts/s3_eval.py (same parser,
tool-result format, 16 turns, 256 tokens) on the CW environment. Greedy by default; --samples k > 0 adds k sampled
episodes per task at T = 1 (vLLM seed fixed), reported separately.

Metrics (per split, per fault class and per write site):
  vsc, r_v, reads (agent read calls per episode), j_c = r_v - c * reads, duplicate, false_success,
  verify_after[step]: share of the step's acknowledged writes followed by a read of that step's record before the
  next write or the report (the selectivity readout: reading after post has no value under the CW prior).

    python -m actionproof.ad.evaluate --model runs/rft_proc/merged --adapter runs/ad_awrb_s0/final --tag ad_awrb_s0 \
        --split cw_dev --out-dir results/ad/dev
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from actionproof.ad.rewards import COST
from actionproof.ad.worlds import make_env_cw, marginalise
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.env import READ_TOOLS, STEP_READ, TOOL_STEP, WRITE_TOOLS
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.stage3.rollout import start_conversation

ROOT = Path(__file__).resolve().parents[2]


def run(llm, tasks, temperature: float = 0.0, seed: int = 0, max_turns: int = 16, max_tokens: int = 256):
    from vllm import SamplingParams
    from actionproof.rollout.episode import _agenttune
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        env = make_env_cw(t, f"cw-{t.task_id}")
        eps.append({"task": t, "env": env, "conv": start_conversation(env, t, PROC_NEUTRAL), "turns": 0, "invalid": 0,
                    "ended_by": "max_turns", "done": False, "n0": len(env.trace)})
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
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                            seed=None if temperature == 0 else seed * 1000 + turn)
        outs = llm.chat([e["conv"] for e in live], sp, use_tqdm=False)
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
        r["ad"] = marginalise(e["task"], e["env"].trace, e["task"].meta["env_seed"])
        r.update(turns=e["turns"], invalid=e["invalid"], ended_by=e["ended_by"], conversation=e["conv"],
                 verify_after=verify_after(r))
        out.append(r)
    return out


def verify_after(rec: dict[str, Any]) -> dict[str, list[int]]:
    """{step: [writes followed by a read of the step's record, acknowledged-ok writes]} for the task PO."""
    pid = rec["task"]["meta"]["po_id"]
    agent = [s for s in rec["trace"] if s.get("by", "agent") == "agent"]
    m: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, s in enumerate(agent):
        if s["tool"] not in TOOL_STEP or s["args"].get("po_id") != pid or s["obs"].get("status") != "ok":
            continue
        step = TOOL_STEP[s["tool"]]
        read = False
        for t in agent[i + 1:]:
            if t["tool"] in WRITE_TOOLS or t["tool"] == "submit_report":
                break
            if t["tool"] == STEP_READ[step] and t["args"].get("po_id") == pid:
                read = True
                break
        m[step][1] += 1
        m[step][0] += read
    return dict(m)


def block(rs: list[dict[str, Any]], cost: float) -> dict[str, Any]:
    n = len(rs)
    g = lambda k: sum(float(r["score"][k]) for r in rs) / n
    reads = sum(r["n_verify"] for r in rs) / n
    va: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rs:
        for s, (a, b) in r["verify_after"].items():
            va[s][0] += a
            va[s][1] += b
    po_reads = sum(sum(x["tool"] == "get_purchase_order" for x in r["trace"] if x.get("by", "agent") == "agent")
                   for r in rs) / n
    return {"n": n, "vsc": g("vsc"), "r_v": g("r_v"), "reads": reads, "j_c": g("r_v") - cost * reads,
            "po_reads": po_reads, "duplicate": g("duplicate_irreversible"), "false_success": g("false_success"),
            "calls": sum(r["n_model_calls"] for r in rs) / n,
            "report_rate": sum(r["ended_by"] == "report" for r in rs) / n,
            "verify_after": {s: (a / b if b else None, b) for s, (a, b) in sorted(va.items())}}


def summarize(recs: list[dict[str, Any]], cost: float = COST) -> dict[str, Any]:
    by = defaultdict(list)
    for r in recs:
        by[r["fault_class"]].append(r)
    faulted = [r for r in recs if r["fault_class"] != "cw:clean"]
    calls = sum(r["n_model_calls"] for r in recs)
    return {**block(recs, cost), "tool_call_validity": 1 - sum(r["invalid"] for r in recs) / max(calls, 1),
            "faulted": block(faulted, cost) if faulted else None,
            "by_class": {k: block(v, cost) for k, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--cost", type=float, default=COST)
    ap.add_argument("--out-dir", default="results/ad/dev")
    a = ap.parse_args()
    from vllm import LLM
    from actionproof.train.merge import merge
    llm = LLM(merge(a.adapter, a.model) if a.adapter else a.model, gpu_memory_utilization=0.85, max_model_len=8192,
              dtype="bfloat16", seed=0)
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for mode in ["greedy"] + [f"t1_s{k}" for k in range(a.samples)]:
        t0 = time.time()
        recs = run(llm, tasks) if mode == "greedy" else run(llm, tasks, 1.0, seed=int(mode.split("_s")[1]))
        suffix = "" if mode == "greedy" else f".{mode}"
        with (out / f"{a.tag}__{a.split}{suffix}.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps({**r, "tag": a.tag, "adapter": a.adapter, "mode": mode}, default=str) + "\n")
        s = {**summarize(recs, a.cost), "tag": a.tag, "split": a.split, "mode": mode, "cost": a.cost,
             "seconds": round(time.time() - t0)}
        (out / f"{a.tag}__{a.split}{suffix}.summary.json").write_text(json.dumps(s, indent=2))
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
                          if k not in ("by_class", "faulted")}))


if __name__ == "__main__":
    main()
