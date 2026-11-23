"""P1 AuditKIT score (PREREGISTRATION_P1.md §6, secondary and descriptive). Never decides an outcome; the verifier does.

**Sample, fixed before any P1 test output (D-P1-1).**
- The greedy episode of seed 0 of each arm (A, B, C, D) on every task of the five test splits: 4 × 450 = 1,800.
- No other selection.

**Judge.**
- AuditKIT's `TaskCompletion` metric, with its exact prompt assembled by `_render` / `_assemble`.
- The input is the agent's own policy text (the system prompt up to the tool list), the ticket, the full trace
  (scripted prefix included) and the submitted report. No database ledger is shown: it is the plain monitor.
- Judge model: Aya Expanse 32B, local vLLM, temperature 0, 400 tokens. FP8 only on GPUs with native FP8 (as in
  `actionproof/eval/judge_audit.py`).

**Reported per arm.**
- Judge acceptance (score = 1).
- Acceptance on verifier failures (VSC = 0, the false-accept rate), and on verifier successes.
- Accuracy against the verifier.

    python scripts/p1_auditkit.py            # needs results/p1/test/{A,B,C,D}_s0__p1_test_*.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "B", "C", "D")
TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
OUT = ROOT / "results/p1/auditkit"


def policy_of(rec: dict) -> str:
    return rec["conversation"][0]["content"].split("\n\nCall one tool per turn")[0]


def load_pool() -> list[dict]:
    pool = []
    for arm in ARMS:
        for sp in TEST:
            f = ROOT / f"results/p1/test/{arm}_s0__{sp}.jsonl"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            for r in map(json.loads, f.read_text().splitlines()):
                if r["sample"] == -1:
                    pool.append({"arm": arm, "split": sp, "task_id": r["task_id"], "vsc": r["vsc"],
                                 "false_success": r["score"]["false_success"], "policy": policy_of(r),
                                 "task": r["task_prompt"], "trace": r["trace"], "report": r["report"]})
    return pool


def summarise(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        ok = [r for r in rs if r["score"] is not None]
        fail = [r for r in ok if not r["vsc"]]
        succ = [r for r in ok if r["vsc"]]
        out[arm] = {"n": len(rs), "unparsed": len(rs) - len(ok),
                    "accept_rate": sum(r["score"] == 1.0 for r in ok) / len(ok) if ok else None,
                    "accept_on_verifier_fail": sum(r["score"] == 1.0 for r in fail) / len(fail) if fail else None,
                    "n_verifier_fail": len(fail),
                    "accept_on_verifier_success": sum(r["score"] == 1.0 for r in succ) / len(succ) if succ else None,
                    "agreement_with_verifier": sum((r["score"] == 1.0) == bool(r["vsc"]) for r in ok) / len(ok) if ok else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/aya-expanse-32b")
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    import torch
    from auditkit.metrics.agent import TaskCompletion
    from auditkit.sample import Sample
    from vllm import LLM, SamplingParams

    from actionproof.rewards.llm_judge import render_report, render_trajectory
    pool = load_pool()
    tc = TaskCompletion(judge_model=object())  # model unused: generation is batched here
    prompts = []
    for p in pool:
        user = tc._render(Sample(input=f"{p['policy']}\n\nTicket:\n{p['task']}"), render_report(p["report"]))
        prompts.append(tc._assemble(user.replace("{trajectory}", render_trajectory(p["trace"]))))
    fp8 = "32b" in a.model.lower() and torch.cuda.get_device_capability() >= (8, 9)
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.9, quantization="fp8" if fp8 else None)
    outs = llm.chat([[{"role": "user", "content": q}] for q in prompts], SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
    rows = []
    for p, o in zip(pool, outs):
        s = tc._parse(o.outputs[0].text)
        rows.append({k: p[k] for k in ("arm", "split", "task_id", "vsc", "false_success")} |
                    {"score": None if s.metadata.get("unknown") else s.value, "fp8": fp8, "text": o.outputs[0].text[-600:]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "taskcompletion_s0_greedy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summ = {"model": a.model, "fp8": fp8, "sample": "seed 0, greedy, all 450 test tasks per arm", "arms": summarise(rows)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
