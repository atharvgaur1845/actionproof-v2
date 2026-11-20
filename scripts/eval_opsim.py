"""A17 evaluation on the OpsSim domains (PREREGISTRATION_A17): the lockstep vLLM loop of `scripts/eval_proc.py`
(same parser, same tool-result format, greedy, 16 turns, 256 tokens per turn) with a domain's environment, system
prompt and verifier. Two modes:
  main  the domain's 120 tasks, closed loop from the user request;
  fork  the A15 cross-domain arm: a scripted prefix up to the domain's first write and its forced authoritative read
        (`fork_prefix`), then the policy continues; the first decisive action is classified by the domain.
Scripted references (oracle, naive, fixed_retry) run on CPU with `--scripted`.

    python scripts/eval_opsim.py --domain care_admin --mode main --scripted --out-dir results/eval/a17
    python scripts/eval_opsim.py --domain it_ops --mode fork --model runs/rft_proc/merged --adapter runs/a12_proc_rv_s0/final --tag rv_s0
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.opsim import DATA, OUT_OF_TAXONOMY, load_tasks
from actionproof.rollout.episode import run_episode, system_prompt

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("it_ops", "care_admin")


def dom(name: str):
    return importlib.import_module(f"actionproof.integrity.opsim.{name}")


def _act(tool: str, params: dict[str, Any]) -> str:
    return json.dumps([{"tool_name": tool, "parameters": params}])


FORK_QS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fork_tasks(D, tasks):
    """care_admin stores forked tasks; it_ops stores base tasks that are forked here (same q values)."""
    if all("fork_q" in t.meta for t in tasks):
        return tasks
    out = []
    for t in tasks:
        for q in FORK_QS:
            f = D.fork_task(t, q)
            f.task_id = f"{t.task_id}-q{q}"
            out.append(f)
    return out


def start(D, task, mode: str):
    """(env, conversation, number of prefix calls) for one task."""
    env = D.make_env(task, f"a17-{task.task_id}")
    if mode == "main":
        return env, [{"role": "system", "content": system_prompt(env, D.SYSTEM)},
                     {"role": "user", "content": task.prompt}], 0
    acts = [_act(t, a) for t, a in D.fork_prefix(task)]
    it = iter(acts)
    conv, st = run_episode(env, lambda c: next(it), D.SYSTEM, task.prompt, max_turns=len(acts))
    assert st.calls == len(acts) and not env.done, task.task_id
    return env, conv, len(env.trace)


def finish(D, task, env, conv, n_prefix: int, extra: dict, mode: str) -> dict:
    rec = D.episode_record(task, env)
    rec.update(conversation=conv, **extra)
    if mode == "fork":
        agent = [s for s in env.trace[n_prefix:] if s.get("by", "agent") == "agent"]
        cat = D.first_decisive(task, agent)
        if cat == "other" and not agent and extra.get("ended_by") == "no_call":
            cat = "invalid"
        branch = task.meta.get("fork_branch") or task.meta.get("fork_evidence")
        rec.update(fork_q=task.meta["fork_q"], fork_branch=branch, first=cat,
                   correct=bool(D.fork_correct(task, cat)), base_task=task.task_id.rsplit("-q", 1)[0])
    return rec


def run_model(llm, D, tasks, mode: str, max_turns: int, max_tokens: int) -> list[dict]:
    from vllm import SamplingParams
    spec = importlib.util.spec_from_file_location("eval_proc", ROOT / "scripts/eval_proc.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    _, _, parse = ep._agenttune()
    eps = []
    for t in tasks:
        env, conv, n = start(D, t, mode)
        eps.append({"task": t, "env": env, "conv": conv, "n": n, "turns": 0, "invalid": 0, "ended_by": "max_turns",
                    "done": False})
    tok = llm.get_tokenizer()
    ctx = llm.llm_engine.model_config.max_model_len
    for _ in range(max_turns):
        for e in eps:
            if not e["done"] and len(tok.encode(tok.apply_chat_template(
                    e["conv"], add_generation_prompt=True, tokenize=False), add_special_tokens=False)) + max_tokens > ctx:
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
    return [finish(D, e["task"], e["env"], e["conv"], e["n"],
                   {"ended_by": e["ended_by"], "turns": e["turns"], "invalid": e["invalid"]}, mode) for e in eps]


def run_scripted(D, tasks, mode: str) -> dict[str, list[dict]]:
    """main: the domain's oracle, naive and fixed_retry. fork: references that continue after the prefix:
    never-read (report completed) and fixed retry (repeat the first write with the same arguments)."""
    out = {}
    for name in (("oracle", "naive", "fixed_retry") if mode == "main" else ("naive", "fixed_retry")):
        recs = []
        for t in tasks:
            env, conv, n = start(D, t, mode)
            if mode == "main":
                getattr(D, name)(env, t)
            else:
                if name == "fixed_retry":
                    tool, args = D.fork_prefix(t)[1]
                    env.call(tool, args)
                env.call("submit_report", {"status": "completed", "summary": "done"})
            recs.append(finish(D, t, env, conv, n, {"ended_by": "report" if env.done else "none", "turns": 0,
                                                    "invalid": 0}, mode))
        out[name] = recs
    return out


def ratio(rs: list[dict], key: str) -> dict:
    n = sum(r["score"][key][0] for r in rs)
    d = sum(r["score"][key][1] for r in rs)
    return {"num": n, "den": d, "rate": n / d if d else None}


def mean(rs: list[dict], key: str) -> float | None:
    v = [r["score"][key] for r in rs if r["score"].get(key) is not None]
    return sum(map(float, v)) / len(v) if v else None


def block(rs: list[dict]) -> dict:
    return {"n": len(rs), "vsc": mean(rs, "vsc"), "false_success": mean(rs, "false_success"),
            "goal_met": mean(rs, "goal_met"), "resolved": mean(rs, "resolved"), "collateral": mean(rs, "collateral"),
            "duplicate": mean(rs, "duplicate"), "correct_escalation": mean(rs, "correct_escalation"),
            "phantom_readback": ratio(rs, "phantom_readback"), "verify_after_write": ratio(rs, "verify_after_write"),
            "report_rate": sum(r.get("ended_by") == "report" for r in rs) / len(rs) if rs else None}


def summarize(recs: list[dict], mode: str) -> dict:
    if mode == "fork":
        by = defaultdict(list)
        for r in recs:
            by[r["fork_q"]].append(r)
        return {"n": len(recs), "by_q": {str(q): {"n": len(v), "correct": sum(r["correct"] for r in v) / len(v),
                                                 "first": dict(__import__("collections").Counter(r["first"] for r in v))}
                                        for q, v in sorted(by.items())}}
    by = defaultdict(list)
    for r in recs:
        by[r["fault_class"]].append(r)
    calls = sum(r["n_model_calls"] for r in recs)
    transfer = [r for r in recs if r["fault_class"] not in OUT_OF_TAXONOMY]
    return {**block(recs), "transfer_classes": block(transfer),
            "out_of_taxonomy": block([r for r in recs if r["fault_class"] in OUT_OF_TAXONOMY]),
            "tool_call_validity": 1 - sum(r.get("invalid", 0) for r in recs) / max(calls, 1),
            "mean_calls": calls / len(recs), "by_fault_class": {k: block(v) for k, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--mode", default="main", choices=["main", "fork"])
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="scripted")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out-dir", default="results/eval/a17")
    ap.add_argument("--split", default=None, help="override the task file (e.g. it_ops_pilot)")
    a = ap.parse_args()
    D = dom(a.domain)
    split = a.split or (a.domain if a.mode == "main" else f"{a.domain}_cf")
    tasks = load_tasks(DATA / f"{split}.jsonl")[: a.n_tasks]
    if a.mode == "fork":
        tasks = fork_tasks(D, tasks)
    out_dir = ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.scripted:
        groups = run_scripted(D, tasks, a.mode)
    else:
        from vllm import LLM
        from actionproof.train.merge import merge
        llm = LLM(merge(a.adapter, a.model) if a.adapter else a.model, gpu_memory_utilization=a.gpu_mem,
                  max_model_len=8192, dtype="bfloat16", seed=0)
        groups = {a.tag: run_model(llm, D, tasks, a.mode, a.max_turns, a.max_tokens)}
    res = {}
    for name, rs in groups.items():
        with (out_dir / f"{name}__{split}.jsonl").open("w") as f:
            for r in rs:
                f.write(json.dumps({**r, "policy": name, "model": a.model, "adapter": a.adapter}) + "\n")
        res[name] = summarize(rs, a.mode)
        (out_dir / f"{name}__{split}.summary.json").write_text(json.dumps({**res[name], "seconds": round(time.time() - t0)},
                                                                          indent=2))
    for name, s in res.items():
        if a.mode == "fork":
            print(name, {q: round(v["correct"], 2) for q, v in s["by_q"].items()})
        else:
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["transfer_classes"].items()
                         if k in ("vsc", "false_success", "phantom_readback", "verify_after_write")})


if __name__ == "__main__":
    main()
