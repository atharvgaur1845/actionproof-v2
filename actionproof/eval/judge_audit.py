"""Judge-fooling audit (D1, upgrade 3): how often do transcript judges accept a run the database
says FAILED?

Pool: claimed-"completed" episodes stratified by (fault, G) from every model we have evaluated,
plus a `visible` flag: after the faulted write, the agent made a verification read, i.e. the
failure was observable in the transcript the judge reads. False acceptance on visible failures
cannot be blamed on missing information.

Judges (all see task + tool calls + responses + report, never the DB):
  - llm_judge  : our transcript judge (rewards/llm_judge.py; also the `llm_judge` reward arm)
  - auditkit_tc: AuditKIT TaskCompletion: its exact prompt, system prompt, CoT instructions and
                 parser (complete=1 / partial=0.5 / failed=0), batched through vLLM.
Acceptance = score 1.0. FAR(fault) = P(accept | G=0); TAR = P(accept | G=1). Wilson 95% CIs.

    python -m actionproof.eval.judge_audit build --out results/judge/pool.jsonl
    python -m actionproof.eval.judge_audit judge --pool results/judge/pool.jsonl --model CohereLabs/aya-expanse-32b --tag aya32b
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from actionproof.env.tools_ecom import VERIFICATION_TOOLS
from actionproof.eval.stats import wilson
from actionproof.rewards.llm_judge import judge_prompt, parse_verdict, render_report, render_trajectory

SOURCES = ["results/teacher/full_32b_verify_a.jsonl", "results/eval/r1/*.jsonl", "results/eval/r3/*.jsonl",
           "results/interp/kl/*/step*.jsonl", "results/eval/pilot_*.jsonl"]


def with_policy(task: str) -> str:
    """Judges get the operator policy too, so policy violations (F5, F7) are judgeable."""
    from actionproof.bench.prompts import POLICY
    return f"{POLICY}\n\nTicket:\n{task}"


def fault_label(e: dict) -> str:
    return "+".join(e.get("faults") or []) or "benign"


def visible(e: dict) -> bool:
    """A verification read happened after the first faulted write attempt."""
    idx = [a["action_idx"] for a in e.get("attempts", []) if a.get("fault")]
    return bool(idx) and any(s["idx"] > min(idx) and s["tool"] in VERIFICATION_TOOLS for s in e["trace"])


def build(out: str, per_cell: int, seed: int) -> None:
    cells = defaultdict(list)
    seen = set()
    for pat in SOURCES:
        for f in sorted(glob.glob(pat)):
            for line in open(f):
                e = json.loads(line)
                if e["verdict"]["reported"] != "completed" or "conversation" not in e:
                    continue
                task = e["conversation"][1]["content"]
                key = (task, json.dumps([s["tool"] for s in e["trace"]]))
                if key in seen:
                    continue
                seen.add(key)
                cells[(fault_label(e), e["verdict"]["G"])].append({
                    "source": f, "task_id": e["task_id"], "fault": fault_label(e), "G": e["verdict"]["G"],
                    "visible": visible(e), "task": task, "trace": e["trace"], "report": e["report"]})
    rng = random.Random(seed)
    pool = []
    for k, v in sorted(cells.items()):
        rng.shuffle(v)
        pool += v[:per_cell]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for p in pool:
            fh.write(json.dumps(p) + "\n")
    print(json.dumps({f"{k[0]}|G={k[1]}": min(len(v), per_cell) for k, v in sorted(cells.items())}))


def auditkit_prompts(pool: list[dict]) -> tuple[list[str], object]:
    from auditkit.metrics.agent import TaskCompletion
    from auditkit.sample import Sample
    tc = TaskCompletion(judge_model=object())  # model unused: we batch generation ourselves
    prompts = []
    for p in pool:
        user = tc._render(Sample(input=with_policy(p["task"])), render_report(p["report"]))
        prompts.append(tc._assemble(user.replace("{trajectory}", render_trajectory(p["trace"]))))
    return prompts, tc


def judge(pool_path: str, model: str, tag: str, out_dir: str, max_tokens: int, tp: int) -> None:
    from vllm import LLM, SamplingParams
    pool = [json.loads(l) for l in open(pool_path)]
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.9, tensor_parallel_size=tp,
              quantization="fp8" if "32b" in model.lower() else None)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    ak_prompts, tc = auditkit_prompts(pool)
    jobs = {"llm_judge": [judge_prompt(with_policy(p["task"]), p["trace"], p["report"]) for p in pool],
            "auditkit_tc": ak_prompts}
    parsers = {"llm_judge": parse_verdict,
               "auditkit_tc": lambda t: (None if (s := tc._parse(t)).metadata.get("unknown") else s.value)}
    res = {}
    for name, prompts in jobs.items():
        outs = llm.chat([[{"role": "user", "content": q}] for q in prompts], sp)
        res[name] = [parsers[name](o.outputs[0].text) for o in outs]
        rows = [{**{k: p[k] for k in ("source", "task_id", "fault", "G", "visible")}, "judge": name, "model": tag,
                 "score": s, "text": o.outputs[0].text[-600:]} for p, s, o in zip(pool, res[name], outs)]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{out_dir}/{tag}__{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps(summarize(out_dir, tag), indent=1))


def summarize(out_dir: str, tag: str) -> dict:
    summ = {}
    for f in sorted(glob.glob(f"{out_dir}/{tag}__*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["score"] is None:
                by[("unparsed", "")][0] += 1
                continue
            acc = r["score"] == 1.0
            for key in ((r["fault"], r["G"]), ("ALL", r["G"]), (f"ALL|visible={r['visible']}", r["G"])):
                by[key][0] += acc
                by[key][1] += 1
        table = {}
        for (fault, G), (a, n) in sorted(by.items(), key=lambda x: str(x[0])):
            if fault == "unparsed":
                table["unparsed"] = a
                continue
            _, lo, hi = wilson(a, n)
            table[f"{fault}|{'FAR' if G == 0 else 'TAR'}"] = {"rate": round(a / n, 3), "n": n, "ci95": [round(lo, 3), round(hi, 3)]}
        summ[Path(f).stem] = table
    Path(out_dir, f"{tag}__summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="results/judge/pool.jsonl")
    b.add_argument("--per-cell", type=int, default=60)
    b.add_argument("--seed", type=int, default=0)
    j = sub.add_parser("judge")
    j.add_argument("--pool", default="results/judge/pool.jsonl")
    j.add_argument("--model", required=True)
    j.add_argument("--tag", required=True)
    j.add_argument("--out-dir", default="results/judge")
    j.add_argument("--max-tokens", type=int, default=400)
    j.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.per_cell, a.seed)
    else:
        judge(a.pool, a.model, a.tag, a.out_dir, a.max_tokens, a.tp)


if __name__ == "__main__":
    main()
