"""A11 secondary: the ledger-INSTRUCTED judge. Same pool, models and judges as the Amendment 9 grounded audit
(`actionproof/eval/judge_audit.py --grounded`), plus one instruction telling the judge that the receipt ledger is
authoritative. Tests whether judges need instructions rather than only evidence (PLAN.md C4 / H4).

    python scripts/judge_instructed.py --model CohereLabs/aya-expanse-32b --tag aya32b_instructed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionproof.eval.judge_audit import attach_attempts, summarize, with_policy
from actionproof.rewards.llm_judge import judge_prompt, parse_verdict, render_ledger, render_report, render_trajectory

INSTRUCTION = ("\n\nIMPORTANT: The receipt ledger above comes from the company database and is authoritative. A write "
               "marked as not committed did not happen, whatever the tool replied. If any write the task requires is not "
               "committed, or only partially committed, the task is NOT complete.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results/judge/pool.jsonl")
    ap.add_argument("--model", default="CohereLabs/aya-expanse-32b")
    ap.add_argument("--tag", default="aya32b_instructed")
    ap.add_argument("--out-dir", default="results/judge")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    import torch
    from auditkit.metrics.agent import TaskCompletion
    from auditkit.sample import Sample
    from vllm import LLM, SamplingParams
    pool = [json.loads(l) for l in open(a.pool)]
    attach_attempts(pool)
    fp8 = "32b" in a.model.lower() and torch.cuda.get_device_capability() >= (8, 9)
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.9, tensor_parallel_size=a.tp,
              quantization="fp8" if fp8 else None)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    tc = TaskCompletion(judge_model=object())
    ak = []
    for p in pool:
        user = tc._render(Sample(input=with_policy(p["task"])), render_report(p["report"]))
        ak.append(tc._assemble(user.replace("{trajectory}", render_trajectory(p["trace"]) + render_ledger(p["attempts"])
                                            + INSTRUCTION)))
    jobs = {"llm_judge": [judge_prompt(with_policy(p["task"]), p["trace"], p["report"], p["attempts"]) + INSTRUCTION
                          for p in pool], "auditkit_tc": ak}
    parsers = {"llm_judge": parse_verdict,
               "auditkit_tc": lambda t: (None if (s := tc._parse(t)).metadata.get("unknown") else s.value)}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for name, prompts in jobs.items():
        outs = llm.chat([[{"role": "user", "content": q}] for q in prompts], sp)
        with open(f"{a.out_dir}/{a.tag}__{name}.jsonl", "w") as fh:
            for p, o in zip(pool, outs):
                fh.write(json.dumps({**{k: p[k] for k in ("source", "task_id", "fault", "G", "visible")}, "judge": name,
                                     "model": a.tag, "fp8": fp8,
                                     "ledger_fail": any(not x["committed"] or x.get("fault") == "F3_partial"
                                                        for x in p["attempts"]),
                                     "score": parsers[name](o.outputs[0].text), "text": o.outputs[0].text[-600:]}) + "\n")
    print(json.dumps(summarize(a.out_dir, a.tag), indent=1))


if __name__ == "__main__":
    main()
