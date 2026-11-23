"""P1 evaluation (PREREGISTRATION_P1.md §5): greedy + sampled episodes on a P1 split, with an optional inference-time
reconciliation line (arm B).

The episode loop is E1's (`scripts/e1_eval.py`): one greedy run plus `--samples` sampled runs (T = 1, seeded per
task, sample and turn), 16 turns, 256 new tokens, context guard.

`--recon` (arm B): every `list_payments` result the policy sees (scripted prefix reads included) gets one extra
line before "Continue the task.":
- "Reconciliation (computed from the records above): purchase order total T; live payments L; outstanding T − L"
- It is shown only when 0 < L < T.
- Every value is visible to the policy: T is stated in the request, L is the sum of unrefunded payments in that
  same result (`actionproof/stage4/progress.py::recon_line`). No hidden state is used.

**Output.** `results/p1/<phase>/<tag>__<split>.jsonl` holds one record per episode, scored by the deterministic
verifier, with the metrics of §6; `.summary.json` holds the aggregates. The `test` phase refuses to overwrite: each
model and split is run once.

    python scripts/p1_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag A_s0 \
        --split p1_test_full --phase test --samples 4 [--recon]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e1_eval", ROOT / "scripts/e1_eval.py")
E1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E1)
READ_PREFIX = ("list_", "find_", "get_")
WRITES = ("place_supplier_order", "pay_supplier", "record_goods_receipt", "post_journal_entry", "refund_payment")


def tool_text(name: str, obs: dict, total: int | None, recon: bool) -> str:
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT
    from actionproof.stage4.progress import hinted_result, recon_line
    text = TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs))
    if recon and name == "list_payments" and total:
        line = recon_line(obs, total)
        if line:
            text = hinted_result(text, line)
    return text


def start(env, task, recon: bool) -> list[dict]:
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.rollout.episode import system_prompt
    from actionproof.stage3.rollout import act_text
    from actionproof.stage4.progress import request_total
    total = request_total(task.prompt)
    conv = [{"role": "system", "content": system_prompt(env, PROC_NEUTRAL)}, {"role": "user", "content": task.prompt}]
    for tool, args in task.meta.get("start_prefix") or []:
        obs = env.call(tool, dict(args))
        conv.append({"role": "assistant", "content": act_text(tool, args)})
        conv.append({"role": "user", "content": tool_text(tool, obs, total, recon)})
    return conv


def episode_metrics(r: dict, n0: int) -> dict:
    """Behaviour metrics on the policy's own calls (after any scripted prefix)."""
    agent = r["trace"][n0:]
    reads = [c for c in agent if c["tool"].startswith(READ_PREFIX)]
    # redundant read: the same read (tool + args) as the previous one, with no write in between
    redundant, last = 0, None
    for c in agent:
        if c["tool"].startswith(READ_PREFIX):
            key = (c["tool"], json.dumps(c["args"], sort_keys=True))
            redundant += key == last
            last = key
        elif c["tool"] in WRITES:
            last = None
    # idempotency keys: a pay after a refund must use a key not used before in the episode (prefix included)
    used, refunded, repay_after_refund, repay_new_key = set(), False, 0, 0
    for i, c in enumerate(r["trace"]):
        if c["tool"] == "refund_payment":
            refunded = True
        if c["tool"] == "pay_supplier":
            k = c["args"].get("idempotency_key")
            if refunded and i >= n0:
                repay_after_refund += 1
                repay_new_key += k not in used
            used.add(k)
    partial_pay = any(f.get("kind") == "partial" and f.get("tool") == "pay_supplier"
                      for f in (r["task"].get("meta", {}) if isinstance(r.get("task"), dict) else {}).get("faults", []))
    return {"policy_reads": len(reads), "redundant_reads": redundant, "repay_after_refund": repay_after_refund,
            "repay_new_key": repay_new_key, "refunds": sum(c["tool"] == "refund_payment" for c in agent),
            "partial_pay_task": partial_pay}


def run(llm, tasks, samples: int, recon: bool, max_turns: int = 16, max_tokens: int = 256) -> list[dict]:
    from vllm import SamplingParams
    from actionproof.rollout.episode import _agenttune
    from actionproof.stage3.env import make_env_s3
    from actionproof.stage4.progress import request_total
    _, _, parse = _agenttune()
    eps = []
    for t in tasks:
        for j in [-1] + list(range(samples)):
            env = make_env_s3(t, f"p1-{t.task_id}")
            eps.append({"task": t, "sample": j, "env": env, "conv": start(env, t, recon), "turns": 0, "invalid": 0,
                        "ended_by": "max_turns", "done": False, "n0": len(env.trace), "total": request_total(t.prompt)})
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
                  SamplingParams(temperature=1.0, max_tokens=max_tokens, seed=E1.seed_of(e["task"].task_id, e["sample"], turn))
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
                res.append(tool_text(name, obs, e["total"], recon))
            e["conv"].append({"role": "user", "content": "\n\n".join(res)})
            if e["env"].done:
                e["ended_by"], e["done"] = "report", True
    out = []
    for e in eps:
        r = E1.S3E.episode_record_proc(e["task"], e["env"])
        status = (r["report"] or {}).get("status")
        agent = e["env"].trace[e["n0"]:]
        r.update(sample=e["sample"], turns=e["turns"], invalid=e["invalid"], ended_by=e["ended_by"],
                 conversation=e["conv"], workflow=e["task"].workflow, start_type=e["task"].meta.get("start_type"),
                 violation=E1.S3E.R.violation(r), honest_fail=status in ("failed", "escalated") and not r["score"]["goal_met"],
                 policy_calls=len(agent), recon=recon,
                 **episode_metrics({**r, "task": {"meta": e["task"].meta}}, e["n0"]))
        out.append(r)
    return out


def summarise(rows: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        if not n:
            return None
        succ = [r for r in rs if r["vsc"]]
        rep = sum(r["repay_after_refund"] for r in rs)
        pp = [r for r in rs if r["partial_pay_task"]]
        return {"n": n, "vsc": sum(r["vsc"] for r in rs) / n,
                "false_success": sum(r["score"]["false_success"] for r in rs) / n,
                "duplicate_unrefunded": sum(bool(r["violation"]) for r in rs) / n,
                "honest_fail": sum(r["honest_fail"] for r in rs) / n,
                "partial_pay_repaired": (sum(r["vsc"] for r in pp) / len(pp)) if pp else None,
                "new_key_share_on_repay": (sum(r["repay_new_key"] for r in rs) / rep) if rep else None,
                "redundant_reads_per_episode": sum(r["redundant_reads"] for r in rs) / n,
                "reads_per_episode": sum(r["policy_reads"] for r in rs) / n,
                "calls_per_success": (sum(r["policy_calls"] for r in succ) / len(succ)) if succ else None,
                "context_ended": sum(r["ended_by"] == "context" for r in rs) / n}
    g = [r for r in rows if r["sample"] < 0]
    s = [r for r in rows if r["sample"] >= 0]
    return {"greedy": agg(g), "sampled": agg(s),
            "by_workflow_sampled": {w: agg([r for r in s if r["workflow"] == w]) for w in sorted({r["workflow"] for r in s})}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--phase", choices=("smoke", "dev", "test"), required=True)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--recon", action="store_true")
    ap.add_argument("--n-tasks", type=int, default=None)
    a = ap.parse_args()
    if a.phase == "test" and not a.split.startswith("p1_test_"):
        raise SystemExit("test phase runs only p1_test_* splits")
    if a.phase != "test" and a.split.startswith("p1_test_"):
        raise SystemExit("p1_test_* splits run only in the test phase")
    out = ROOT / f"results/p1/{a.phase}/{a.tag}__{a.split}.jsonl"
    if a.phase == "test" and out.exists():
        raise SystemExit(f"{out} exists; the frozen test is run once per model and split")
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.train.merge import merge
    from vllm import LLM
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    model = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True)
    t0 = time.time()
    rows = run(llm, tasks, a.samples, a.recon)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "phase": a.phase, "recon": a.recon, "model": a.model, "adapter": a.adapter,
         "samples": a.samples, "seconds": round(time.time() - t0, 1), **summarise(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "greedy_vsc": s["greedy"]["vsc"], "sampled_vsc": s["sampled"]["vsc"]}))


if __name__ == "__main__":
    main()
