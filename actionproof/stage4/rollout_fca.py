"""FCA rollouts (docs/research/algorithm/FCA_GRPO.md §2). Same lockstep loop as actionproof/stage3/rollout.py; for a
decision-state task (a recovery start of type ack/read), the last G - M slots of its group of identical prompts are
forced branches: the first assistant turn is the canonical action of one class (actionproof/stage4/forks.py), logged
with sampling log-probability 0 so TRL's sequence-level importance correction weights the branch by pi_old(action);
the rest of the episode is sampled. Per-completion fork metadata is stored on the trainer for the advantage step.
"""
from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from typing import Any, Callable

from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.env import READ_TOOLS
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.rollout.agenttune_rollout import prompt_key
from actionproof.rollout.episode import _agenttune
from actionproof.stage3.env import make_env_s3
from actionproof.stage3.rollout import ASSISTANT_TAIL, act_text, start_conversation
from actionproof.stage4 import forks

N_ON = 4


def make_rollout_func(tasks: list[Any], base_prompt: str, max_turns: int = 16, max_tokens: int = 256,
                      temperature: float = 1.0, max_context: int = 7168, n_on: int = N_ON) -> Callable:
    by_prompt = {t.prompt: t for t in tasks}
    assert len(by_prompt) == len(tasks)
    _, _, parse = _agenttune()

    def render(tok, conv, gen):
        return tok.apply_chat_template(conv, add_generation_prompt=gen, tokenize=False)

    def execute(e, text, tok, before):
        calls = parse(text) or []
        if not calls:
            e["done"] = True
            return
        results = []
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = None
            obs = e["env"].call(name, args) if isinstance(args, dict) else {"status": "error", "error": "invalid_arguments"}
            results.append(TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs)))
        e["conv"].append({"role": "user", "content": "\n\n".join(results)})
        if e["env"].done:
            e["done"] = True
            return
        after = render(tok, e["conv"], True)
        if not after.startswith(before):
            raise RuntimeError("chat template lost the prefix property")
        obs_ids = tok.encode(after[len(before):], add_special_tokens=False)
        if len(e["prompt_ids"]) + len(e["ids"]) + len(obs_ids) + max_tokens > max_context:
            e["done"] = True
            return
        e["ids"] += obs_ids
        e["mask"] += [0] * len(obs_ids)
        e["lps"] += [0.0] * len(obs_ids)

    def rollout_func(prompts: list[Any], trainer: Any = None, **_: Any) -> dict[str, Any]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        tok = trainer.processing_class
        llm = trainer.vllm_generation.llm
        tail_ids = tok.encode(ASSISTANT_TAIL, add_special_tokens=False)
        stop_ids = sorted({tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>"), tok.convert_tokens_to_ids("<|END_RESPONSE|>")})
        seen: Counter = Counter()
        eps = []
        step = getattr(getattr(trainer, "state", None), "global_step", 0)
        for p in prompts:
            key = prompt_key(p)
            task = by_prompt[key]
            slot = seen[key]
            seen[key] += 1
            rid = f"ro-{uuid.uuid4().hex[:12]}"
            env = make_env_s3(task, rid, seed=uuid.UUID(rid[3:].ljust(32, "0")).int % 2**31)
            conv = start_conversation(env, task, base_prompt)
            pids = tok.encode(render(tok, conv, True), add_special_tokens=False)
            e = {"task": task, "env": env, "conv": conv, "prompt_ids": list(pids), "ids": [], "mask": [], "lps": [],
                 "done": env.done, "n0": len(env.trace),
                 "meta": {"group": f"{step}:{key[:60]}", "kind": "plain", "cls": None, "span": None}}
            classes = forks.available(task, conv)
            if classes:
                e["meta"]["kind"] = "on"
                if slot >= n_on:
                    cls = classes[(slot - n_on) % len(classes)]
                    tool, args = forks.forced_action(task, conv, cls)
                    text = act_text(tool, args)
                    gen = tok.encode(text, add_special_tokens=False)
                    s = len(e["ids"])
                    e["ids"] += gen + tail_ids
                    e["mask"] += [1] * len(gen) + [0] * len(tail_ids)
                    e["lps"] += [0.0] * len(gen) + [0.0] * len(tail_ids)  # forced: mu = 1, so the IS ratio is pi_old
                    e["meta"].update(kind="forced", cls=cls, span=(s, s + len(gen)))
                    before = render(tok, e["conv"] + [{"role": "assistant", "content": text}], False)
                    e["conv"].append({"role": "assistant", "content": text})
                    execute(e, text, tok, before)
            eps.append(e)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=0, stop_token_ids=stop_ids)
        for _turn in range(max_turns):
            live = [e for e in eps if not e["done"]]
            if not live:
                break
            outs = llm.generate([TokensPrompt(prompt_token_ids=e["prompt_ids"] + e["ids"]) for e in live],
                                sampling_params=sp, use_tqdm=False)
            for e, o in zip(live, outs):
                c = o.outputs[0]
                gen = list(c.token_ids)
                lps = [float(d[t].logprob) if d and t in d else 0.0 for t, d in zip(gen, c.logprobs or [{}] * len(gen))]
                while gen and gen[-1] in stop_ids:
                    gen.pop(); lps.pop()
                text = tok.decode(gen, skip_special_tokens=True)
                e["ids"] += gen + tail_ids
                e["mask"] += [1] * len(gen) + [0] * len(tail_ids)
                e["lps"] += lps + [0.0] * len(tail_ids)
                before = render(tok, e["conv"] + [{"role": "assistant", "content": text}], False)
                e["conv"].append({"role": "assistant", "content": text})
                execute(e, text, tok, before)
        records = [episode_record_proc(e["task"], e["env"]) for e in eps]
        for e in eps:  # pool on-policy rollouts into Q estimates by the class of their first policy call
            if e["meta"]["kind"] == "on":
                first = [x for x in e["env"].trace[e["n0"]:] if x.get("by", "agent") == "agent"]
                if e["task"].meta.get("start_type") == "read":
                    first = [x for x in first if x["tool"] not in READ_TOOLS]
                if first:
                    e["meta"]["cls"] = forks.classify(e["task"], first[0]["tool"], first[0]["args"] or {})
        trainer._fca_meta = [e["meta"] for e in eps]
        trainer._fca_rewards = [float(r["score"]["r_v"]) for r in records]
        log = os.environ.get("AP_ROLLOUT_LOG")
        if log:
            with open(log, "a") as f:
                for e, r in zip(eps, records):
                    f.write(json.dumps({"step": step, "fca": e["meta"], "start_type": e["task"].meta.get("start_type"),
                                        **r}, default=str) + "\n")
        return {"prompt_ids": [e["prompt_ids"] for e in eps], "completion_ids": [e["ids"] for e in eps],
                "logprobs": [e["lps"] for e in eps], "env_mask": [e["mask"] for e in eps], "ap_episodes": records,
                "tool_call_counts": [len(e["env"].trace) for e in eps]}

    return rollout_func
