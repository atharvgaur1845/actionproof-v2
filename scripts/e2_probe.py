"""E2 (PREREGISTRATION_E1E2): the next decisive action at fixed decision contexts, by workflow position.

For each `e2_pp` task (a read start: the prefix ends with the authoritative `list_payments` read that shows the partial
payment), four contexts are built in the task's own Stage 3 environment, exactly as `scripts/s3_eval.py` starts an
episode:
  S1V0  the read start as is (the evidence of the partial payment is the last tool result)
  S1V1  the same, with one reconciliation line appended to that tool result, computed only from values visible in the
        conversation (the PO total from the request and the payment amounts in the read)
  S2V0  S1V0 plus the scripted correct refund of the partial payment and its acknowledgement
  S2V1  S2V0 with the reconciliation line after the refund (live payments 0, outstanding = PO total)
One assistant turn is generated per context: greedy once, and 16 samples at temperature 1 (reproducible seed). The
first tool call is classified; no further turns are run.
    python scripts/e2_probe.py --model <merged model dir> --tag s3_rv_mix_s0
Output: results/e2/<tag>.jsonl (one record per context x sample) + <tag>.summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.agents import _po
from actionproof.integrity.sim.prompts import PROC_NEUTRAL
from actionproof.stage3.env import make_env_s3
from actionproof.stage3.rollout import act_text, start_conversation

ROOT = Path(__file__).resolve().parents[1]
TAIL = "\nContinue the task."
CONDS = ("S1V0", "S1V1", "S2V0", "S2V1")
K = 16


def recon(content: str, total: int, live: int) -> str:
    line = (f"\nReconciliation (computed from the records above): purchase order total {total} cents; live payments "
            f"{live} cents; outstanding {total - live} cents.")
    assert content.count(TAIL) == 1
    return content.replace(TAIL, line + TAIL)


def contexts(task) -> dict[str, dict]:
    """Four decision contexts for one read-start task, plus what is needed to classify the next action."""
    po = _po(task)
    out = {}
    for cond in CONDS:
        env = make_env_s3(task, f"s3-{task.task_id}")
        conv = start_conversation(env, task, PROC_NEUTRAL)
        last = conv[-1]["content"]
        obs = json.loads(re.match(r"Tool result for list_payments: (.*)" + re.escape(TAIL), last, re.S).group(1))
        pays = [p for p in obs["payments"] if p["kind"] == "payment"]
        assert len(pays) == 1 and pays[0]["amount_cents"] < po["amount_cents"], task.task_id
        pid, paid = pays[0]["payment_id"], pays[0]["amount_cents"]
        if cond.startswith("S1"):
            if cond == "S1V1":
                conv[-1]["content"] = recon(last, po["amount_cents"], paid)
        else:
            r = env.call("refund_payment", {"payment_id": pid})
            conv.append({"role": "assistant", "content": act_text("refund_payment", {"payment_id": pid})})
            content = TOOL_RESULT_FORMAT.format(name="refund_payment", content=json.dumps(r))
            conv.append({"role": "user", "content": recon(content, po["amount_cents"], 0) if cond == "S2V1" else content})
        out[cond] = {"conv": conv, "pid": pid, "paid": paid, "total": po["amount_cents"],
                     "key": next(a for t, a in task.meta["start_prefix"] if t == "pay_supplier")["idempotency_key"]}
    return out


def classify(calls: list[dict], c: dict) -> str:
    if not calls:
        return "no_call"
    name = calls[0]["function"]["name"]
    try:
        args = json.loads(calls[0]["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return "invalid"
    if not isinstance(args, dict):
        return "invalid"
    if name == "refund_payment":
        return "refund_ok" if args.get("payment_id") == c["pid"] else "refund_other"
    if name == "pay_supplier":
        amt, key = args.get("amount_cents"), args.get("idempotency_key")
        if amt == c["total"]:
            return "pay_full_newkey" if key != c["key"] else "pay_full_samekey"
        if amt == c["total"] - c["paid"]:
            return "pay_remainder"
        return "pay_other"
    if name == "post_journal_entry":
        return "post"
    if name == "escalate_to_human":
        return "escalate"
    if name == "submit_report":
        return "report_" + str(args.get("status"))
    if name.startswith(("list_", "find_", "get_")):
        return "read"
    return "other:" + name


# registered correctness of the first decisive action (PREREGISTRATION_E1E2 §3)
CORRECT = {"S1": {"refund_ok", "escalate"}, "S2": {"pay_full_newkey", "escalate"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--out-dir", default="results/e2")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from actionproof.rollout.episode import _agenttune
    _, _, parse = _agenttune()
    llm = LLM(a.model, gpu_memory_utilization=0.85, max_model_len=8192, dtype="bfloat16", seed=0)
    tasks = load_tasks(DATA / "e2_pp.jsonl")[: a.n_tasks]
    t0 = time.time()
    items = [(t, cond, c) for t in tasks for cond, c in contexts(t).items()]
    convs = [c["conv"] for _, _, c in items]
    greedy = llm.chat(convs, SamplingParams(temperature=0, max_tokens=256), use_tqdm=False)
    sampled = llm.chat(convs, [SamplingParams(n=K, temperature=1.0, max_tokens=256,
                                              seed=zlib.crc32(f"{t.task_id}|{cond}".encode()) % 2**31)
                               for t, cond, _ in items], use_tqdm=False)
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rate = defaultdict(lambda: [0, 0])
    acts = defaultdict(Counter)
    with (out / f"{a.tag}.jsonl").open("w") as f:
        for (t, cond, c), g, s in zip(items, greedy, sampled):
            for j, o in [(-1, g.outputs[0])] + list(enumerate(s.outputs)):
                act = classify(parse(o.text) or [], c)
                ok = act in CORRECT[cond[:2]]
                f.write(json.dumps({"task_id": t.task_id, "workflow": t.workflow, "ratio": t.meta["faults"][0]["ratio"],
                                    "cond": cond, "sample": j, "action": act, "correct": ok, "text": o.text}) + "\n")
                if j >= 0:
                    rate[(cond, t.workflow)][0] += ok
                    rate[(cond, t.workflow)][1] += 1
                    acts[(cond, t.workflow)][act] += 1
    summ = {"tag": a.tag, "n_tasks": len(tasks), "samples": K, "seconds": round(time.time() - t0),
            "sampled_correct": {f"{c}|{w}": v[0] / v[1] for (c, w), v in sorted(rate.items())},
            "sampled_actions": {f"{c}|{w}": dict(v) for (c, w), v in sorted(acts.items())}}
    (out / f"{a.tag}.summary.json").write_text(json.dumps(summ, indent=2))
    print(a.tag, json.dumps({k: round(v, 3) for k, v in summ["sampled_correct"].items()}))


if __name__ == "__main__":
    main()
