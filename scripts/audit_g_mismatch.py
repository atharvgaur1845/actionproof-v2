"""Audit (docs/quality/EXPERIMENT_PIPELINE_AUDIT.md §D): how far G's sampler (hinted context) was from the
distribution its loss assumed (un-hinted context).

**Why an audit is needed.** G (`actionproof/stage4/rollout_c5g.py`, mode "guide"):
- samples a hinted rollout's tokens from q = π(· | hinted context);
- overwrites their sampling log-probabilities with log p = log π(· | un-hinted context);
- hands those to TRL as if p had sampled them.

TRL's importance ratio is then 1 by construction. The gradient is E_q[A ∇ log p], not E_p[A ∇ log p], so the update
is biased by however far q is from p. Because the originals were overwritten, the rollout logs cannot show the gap.

**Method (an approximation, labelled as such).**
- For every hinted episode in a G rollout log, rebuild both contexts from the trace. Each policy turn is written
  in the canonical action format (`act_text`); the policy's raw text was not logged.
- The tool results are rebuilt exactly. The hint is the same `recon_line` G used.
- Under a G checkpoint, score every assistant token under both contexts and report:
  - the mean per-token log-ratio log q − log p, a Monte-Carlo estimate of KL(q ‖ p) on q's own tokens;
  - the per-episode sum;
  - the effective sample size of the implied importance weights p/q.

    python scripts/audit_g_mismatch.py --log results/s5/rollouts/s5_guide_s0.jsonl.gz \
        --model runs/rft_proc/merged --adapter runs/s5_guide_s0/checkpoint-100
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def contexts(r: dict) -> tuple[list[dict], list[dict], list[int]]:
    """(plain conversation, hinted conversation, indices of assistant turns) rebuilt from one logged episode."""
    from actionproof.bench.prompts import TOOL_RESULT_FORMAT
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.rollout.episode import system_prompt
    from actionproof.stage3.env import make_env_s3
    from actionproof.stage3.rollout import act_text
    from actionproof.stage4.progress import hinted_result, recon_line, request_total
    from actionproof.bench.generator_proc import ProcTask
    task = ProcTask.from_json(r["task"])
    env = make_env_s3(task, "audit")
    sysmsg = {"role": "system", "content": system_prompt(env, PROC_NEUTRAL)}
    plain, hint = [sysmsg, {"role": "user", "content": r["task_prompt"]}], [sysmsg, {"role": "user", "content": r["task_prompt"]}]
    total = request_total(r["task_prompt"])
    n_prefix = len(r["task"]["meta"].get("start_prefix") or [])
    turns = []
    for i, s in enumerate(r["trace"]):
        a = {"role": "assistant", "content": act_text(s["tool"], s["args"])}
        if i >= n_prefix:
            turns.append(len(plain))
        plain.append(a)
        hint.append(a)
        if s["tool"] == "submit_report":
            break
        txt = TOOL_RESULT_FORMAT.format(name=s["tool"], content=json.dumps(s["obs"]))
        plain.append({"role": "user", "content": txt})
        line = recon_line(s["obs"], total) if s["tool"] == "list_payments" and total else None
        hint.append({"role": "user", "content": hinted_result(txt, line) if line else txt})
    return plain, hint, turns


def token_logprobs(llm, tok, conv: list[dict], turn_idx: list[int]) -> list[float]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    out = []
    for k in turn_idx:
        pre = tok.encode(tok.apply_chat_template(conv[:k], add_generation_prompt=True, tokenize=False), add_special_tokens=False)
        full = tok.encode(tok.apply_chat_template(conv[:k + 1], tokenize=False), add_special_tokens=False)
        o = llm.generate([TokensPrompt(prompt_token_ids=full)], SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
        pl = o.prompt_logprobs
        out += [float(pl[t][full[t]].logprob) for t in range(len(pre), len(full)) if pl[t] and full[t] in pl[t]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default="results/quality/g_mismatch.json")
    a = ap.parse_args()
    from vllm import LLM
    from actionproof.train.merge import merge
    eps = [r for r in map(json.loads, gzip.open(ROOT / a.log, "rt")) if r.get("hinted")]
    llm = LLM(model=merge(a.adapter, a.model), max_model_len=8192, gpu_memory_utilization=0.85, seed=0)
    tok = llm.get_tokenizer()
    per_ep, all_d = [], []
    for r in eps:
        plain, hint, turns = contexts(r)
        lq, lp = token_logprobs(llm, tok, hint, turns), token_logprobs(llm, tok, plain, turns)
        n = min(len(lq), len(lp))
        d = [lq[i] - lp[i] for i in range(n)]
        all_d += d
        per_ep.append(sum(d))
    w = [math.exp(-s) for s in per_ep]  # importance weight p/q per episode
    ess = (sum(w) ** 2 / sum(x * x for x in w)) if w else None
    res = {"log": a.log, "adapter": a.adapter, "hinted_episodes": len(eps), "tokens": len(all_d),
           "mean_token_log_ratio_q_over_p": sum(all_d) / len(all_d) if all_d else None,
           "mean_episode_log_ratio": sum(per_ep) / len(per_ep) if per_ep else None,
           "max_episode_log_ratio": max(per_ep) if per_ep else None,
           "importance_ess": ess, "importance_ess_share": (ess / len(w)) if w else None,
           "approximation": "assistant turns rebuilt in canonical act_text format; raw policy text was not logged"}
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
