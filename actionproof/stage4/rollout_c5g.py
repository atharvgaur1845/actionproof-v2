"""Rollout for the Stage 5 development candidates (docs/research/algorithm/C5G_DEV.md).

A copy of actionproof/stage3/rollout.py (unchanged there; same replay, parser, masks, limits) with two additions:
- every episode records each policy turn's token span and the verified progress u = Phi(after) - Phi(before)
  (actionproof/stage4/progress.py), handed to the trainer as `trainer._c5_meta`;
- mode "guide": in every prompt group, rollouts with an odd index generate from a context in which each
  `list_payments` result showing a shortfall carries the reconciliation line. The training sequence is the un-hinted
  conversation. The hinted episode's policy tokens are then re-scored under the un-hinted context (one prefill), and
  those log-probabilities are returned as the sampling log-probabilities, so TRL treats the tokens as on-policy for
  the un-hinted context (PSP-style hint in the sampler only; plain GRPO loss).
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable

from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.rollout.episode import _agenttune
from actionproof.stage3.env import make_env_s3
from actionproof.stage3.rollout import ASSISTANT_TAIL, prompt_key, start_conversation
from actionproof.stage4.progress import hinted_result, phi, recon_line, request_total


def make_rollout_func(tasks: list[Any], base_prompt: str, mode: str, max_turns: int = 16, max_tokens: int = 256,
                      temperature: float = 1.0, max_context: int = 7168) -> Callable:
    assert mode in ("c5", "guide")
    by_prompt = {t.prompt: t for t in tasks}
    assert len(by_prompt) == len(tasks), "prompts must be unique per task"
    _, _, parse = _agenttune()

    def render(tok, conv, gen):
        return tok.apply_chat_template(conv, add_generation_prompt=gen, tokenize=False)

    def rollout_func(prompts: list[Any], trainer: Any = None, **_: Any) -> dict[str, Any]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        tok = trainer.processing_class
        llm = trainer.vllm_generation.llm
        tail_ids = tok.encode(ASSISTANT_TAIL, add_special_tokens=False)
        stop_ids = sorted({tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>"), tok.convert_tokens_to_ids("<|END_RESPONSE|>")})
        eps, seen = [], {}
        for p in prompts:
            task = by_prompt[prompt_key(p)]
            j = seen.get(task.task_id, 0)
            seen[task.task_id] = j + 1
            rid = f"ro-{uuid.uuid4().hex[:12]}"
            env = make_env_s3(task, rid, seed=uuid.UUID(rid[3:].ljust(32, "0")).int % 2**31)
            conv = start_conversation(env, task, base_prompt)
            pids = tok.encode(render(tok, conv, True), add_special_tokens=False)
            tj = task.to_json()
            eps.append({"task": task, "tj": tj, "env": env, "conv": conv, "gconv": [dict(m) for m in conv],
                        "prompt_ids": list(pids), "ids": [], "gids": [], "mask": [], "lps": [], "done": env.done,
                        "hint": mode == "guide" and j % 2 == 1, "hinted": False, "total": request_total(task.prompt),
                        "phi": phi(tj, env.state), "turns": [], "group": task.task_id})
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=0, stop_token_ids=stop_ids)
        for _turn in range(max_turns):
            live = [e for e in eps if not e["done"]]
            if not live:
                break
            outs = llm.generate([TokensPrompt(prompt_token_ids=e["prompt_ids"] + e["gids"]) for e in live],
                                sampling_params=sp, use_tqdm=False)
            for e, o in zip(live, outs):
                c = o.outputs[0]
                gen = list(c.token_ids)
                lps = [float(d[t].logprob) if d and t in d else 0.0 for t, d in zip(gen, c.logprobs or [{}] * len(gen))]
                while gen and gen[-1] in stop_ids:
                    gen.pop(); lps.pop()
                text = tok.decode(gen, skip_special_tokens=True)
                start = len(e["ids"])
                e["ids"] += gen + tail_ids
                e["gids"] += gen + tail_ids
                e["mask"] += [1] * len(gen) + [0] * len(tail_ids)
                e["lps"] += lps + [0.0] * len(tail_ids)
                before = render(tok, e["conv"] + [{"role": "assistant", "content": text}], False)
                gbefore = render(tok, e["gconv"] + [{"role": "assistant", "content": text}], False)
                e["conv"].append({"role": "assistant", "content": text})
                e["gconv"].append({"role": "assistant", "content": text})
                calls = parse(text) or []
                if not calls:
                    e["turns"].append((start, start + len(gen), 0.0))
                    e["done"] = True
                    continue
                results, gresults = [], []
                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = None
                    obs = e["env"].call(name, args) if isinstance(args, dict) else {"status": "error", "error": "invalid_arguments"}
                    r = TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs))
                    results.append(r)
                    line = recon_line(obs, e["total"]) if e["hint"] and name == "list_payments" and e["total"] else None
                    if line:
                        e["hinted"] = True
                    gresults.append(hinted_result(r, line) if line else r)
                new_phi = phi(e["tj"], e["env"].state)
                e["turns"].append((start, start + len(gen), new_phi - e["phi"]))
                e["phi"] = new_phi
                e["conv"].append({"role": "user", "content": "\n\n".join(results)})
                e["gconv"].append({"role": "user", "content": "\n\n".join(gresults)})
                if e["env"].done:
                    e["done"] = True
                    continue
                after, gafter = render(tok, e["conv"], True), render(tok, e["gconv"], True)
                if not after.startswith(before) or not gafter.startswith(gbefore):
                    raise RuntimeError("chat template lost the prefix property")
                obs_ids = tok.encode(after[len(before):], add_special_tokens=False)
                gobs_ids = tok.encode(gafter[len(gbefore):], add_special_tokens=False)
                if len(e["prompt_ids"]) + max(len(e["ids"]) + len(obs_ids), len(e["gids"]) + len(gobs_ids)) + max_tokens > max_context:
                    e["done"] = True
                    continue
                e["ids"] += obs_ids
                e["gids"] += gobs_ids
                e["mask"] += [0] * len(obs_ids)
                e["lps"] += [0.0] * len(obs_ids)
        hinted = [e for e in eps if e["hinted"]]
        if hinted:  # re-score hinted episodes' tokens under the un-hinted context (sampler-only hint)
            outs = llm.generate([TokensPrompt(prompt_token_ids=e["prompt_ids"] + e["ids"]) for e in hinted],
                                sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
            for e, o in zip(hinted, outs):
                pl = o.prompt_logprobs[len(e["prompt_ids"]):]
                for t, (tid, d) in enumerate(zip(e["ids"], pl)):
                    if e["mask"][t]:
                        e["lps"][t] = float(d[tid].logprob) if d and tid in d else e["lps"][t]
        records = [episode_record_proc(e["task"], e["env"]) for e in eps]
        trainer._c5_meta = [{"group": e["group"], "turns": e["turns"], "hinted": e["hinted"]} for e in eps]
        log = os.environ.get("AP_ROLLOUT_LOG")
        if log:
            step = getattr(getattr(trainer, "state", None), "global_step", None)
            with open(log, "a") as f:
                for e, r in zip(eps, records):
                    f.write(json.dumps({"step": step, "start_type": r["task"]["meta"].get("start_type"),
                                        "hinted": e["hinted"], "turn_progress": [u for _, _, u in e["turns"]], **r},
                                       default=str) + "\n")
        return {"prompt_ids": [e["prompt_ids"] for e in eps], "completion_ids": [e["ids"] for e in eps],
                "logprobs": [e["lps"] for e in eps], "env_mask": [e["mask"] for e in eps], "ap_episodes": records,
                "tool_call_counts": [len(e["env"].trace) for e in eps]}

    return rollout_func
