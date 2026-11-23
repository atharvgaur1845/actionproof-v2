"""CPU dry run of the AD rollout, rewards and evaluation loop with a scripted fake LLM (real tokenizer, real parser,
real environment). Protects the shared GPU queue from plumbing failures. Not an experiment.

    HF_HUB_OFFLINE=1 python scripts/ad/cpu_dryrun.py
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

from transformers import AutoTokenizer

from actionproof.ad.evaluate import run as eval_run, summarize
from actionproof.ad.rewards import ESTIMATORS, METRICS, make_reward
from actionproof.ad.rollout import make_rollout_func
from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.integrity.sim.agents import _args, _po
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL
from actionproof.integrity.sim.prompts import PROC_NEUTRAL

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
tasks = load_tasks(DATA / "cw_train.jsonl")[:4]
by_prompt_tokens = {}


def plan(task, read: bool):
    po = _po(task)
    out = []
    for s in task.meta["requested"]:
        out.append((STEP_TOOL[s], _args(s, po, f"{s}-{po['po_id']}")))
        if read:
            out.append((STEP_READ[s], {"po_id": po["po_id"]}))
    out.append(("submit_report", {"status": "completed", "summary": "done"}))
    return [json.dumps([{"tool_name": t, "parameters": a}]) for t, a in out]


class FakeLLM:
    """Returns the next scripted call for whichever task the prompt belongs to (matched by the PO id in the prompt)."""
    def __init__(self, reads):
        self.reads = reads
        self.cursor = {}

    def _task_of(self, text):
        return next(t for t in tasks if t.meta["po_id"] in text)

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        outs = []
        for i, p in enumerate(prompts):
            ids = p["prompt_token_ids"] if isinstance(p, dict) else p.prompt_token_ids
            text = tok.decode(ids)
            t = self._task_of(text)
            k = (id(p), t.task_id)
            n_assist = text.count("<|CHATBOT_TOKEN|>") - 1
            read = self.reads[(hash(text[:200]) + i) % len(self.reads)]
            steps = plan(t, read)
            reply = steps[min(n_assist, len(steps) - 1)]
            gen = tok.encode(reply, add_special_tokens=False)
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=gen, logprobs=[{g: SimpleNamespace(logprob=-0.1)} for g in gen])]))
        return outs


def main():
    rf = make_rollout_func(tasks, PROC_NEUTRAL)
    trainer = SimpleNamespace(processing_class=tok, vllm_generation=SimpleNamespace(llm=FakeLLM([True, False])),
                              state=SimpleNamespace(global_step=0))
    prompts = [[{"role": "system", "content": PROC_NEUTRAL}, {"role": "user", "content": t.prompt}] for t in tasks for _ in range(2)]
    out = rf(prompts, trainer=trainer)
    eps = out["ap_episodes"]
    for e in eps:
        print(e["task_id"], e["fault_class"], "reads", e["n_verify"], "r_v", e["score"]["r_v"], "ad", e["ad"])
        assert e["ad"]["r_bar"] is not None
    for est, (field, scale) in ESTIMATORS.items():
        print(est, scale, [round(x, 3) for x in make_reward(field)(None, [0] * len(eps), ap_episodes=eps)])
    for m in METRICS:
        m(None, [0] * len(eps), ap_episodes=eps)
    print("DRYRUN_ROLLOUT_OK")


if __name__ == "__main__":
    main()
