"""Rollout for candidate D3 (uninformative-read local credit). actionproof/ad/rollout.py plus per-turn bookkeeping:
for every policy turn, its token span in the completion and the trace indices of the calls it made. After the episode,
`actionproof/ad/readinfo.py:read_flags` marks each read as informative or not. A turn is an *uninformative-read turn*
iff every call it made is a read and none of them is informative. The spans go to the trainer as `trainer._ad_meta`.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable

from actionproof.ad.readinfo import read_flags
from actionproof.ad.worlds import make_env_cw, marginalise
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.rollout.agenttune_rollout import prompt_key
from actionproof.rollout.episode import _agenttune
from actionproof.stage3.rollout import ASSISTANT_TAIL, start_conversation


def uninformative_turns(turns: list[tuple[int, int, list[int]]], flags: list[tuple[int, str, bool]]) -> list[tuple[int, int]]:
    """Spans (start, end) of turns whose calls are all uninformative reads."""
    info = {i: ok for i, _, ok in flags}
    out = []
    for s, e, idxs in turns:
        if idxs and all(i in info and not info[i] for i in idxs):
            out.append((s, e))
    return out


def make_rollout_func(tasks: list[Any], base_prompt: str, max_turns: int = 16, max_tokens: int = 256,
                      temperature: float = 1.0, max_context: int = 7168) -> Callable:
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
        eps = []
        for p in prompts:
            task = by_prompt[prompt_key(p)]
            rid = f"ro-{uuid.uuid4().hex[:12]}"
            seed = uuid.UUID(rid[3:].ljust(32, "0")).int % 2**31
            env = make_env_cw(task, rid, seed=seed)
            conv = start_conversation(env, task, base_prompt)
            pids = tok.encode(render(tok, conv, True), add_special_tokens=False)
            eps.append({"task": task, "env": env, "seed": seed, "conv": conv, "prompt_ids": list(pids), "ids": [],
                        "mask": [], "lps": [], "done": env.done, "turns": []})
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
                start = len(e["ids"])
                e["ids"] += gen + tail_ids
                e["mask"] += [1] * len(gen) + [0] * len(tail_ids)
                e["lps"] += lps + [0.0] * len(tail_ids)
                before = render(tok, e["conv"] + [{"role": "assistant", "content": text}], False)
                e["conv"].append({"role": "assistant", "content": text})
                calls = parse(text) or []
                if not calls:
                    e["turns"].append((start, start + len(gen), []))
                    e["done"] = True
                    continue
                results, idxs = [], []
                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = None
                    if isinstance(args, dict):
                        idxs.append(len(e["env"].trace))
                        obs = e["env"].call(name, args)
                    else:
                        obs = {"status": "error", "error": "invalid_arguments"}
                    results.append(TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs)))
                e["turns"].append((start, start + len(gen), idxs))
                e["conv"].append({"role": "user", "content": "\n\n".join(results)})
                if e["env"].done:
                    e["done"] = True
                    continue
                after = render(tok, e["conv"], True)
                if not after.startswith(before):
                    raise RuntimeError("chat template lost the prefix property")
                obs_ids = tok.encode(after[len(before):], add_special_tokens=False)
                if len(e["prompt_ids"]) + len(e["ids"]) + len(obs_ids) + max_tokens > max_context:
                    e["done"] = True
                    continue
                e["ids"] += obs_ids
                e["mask"] += [0] * len(obs_ids)
                e["lps"] += [0.0] * len(obs_ids)
        records, meta = [], []
        for e in eps:
            r = episode_record_proc(e["task"], e["env"])
            r["ad"] = marginalise(e["task"], e["env"].trace, e["seed"])
            fl = read_flags(e["task"], e["env"].trace, e["seed"])
            r["n_inf_reads"] = sum(ok for _, _, ok in fl)
            r["n_uninf_reads"] = sum(not ok for _, _, ok in fl)
            spans = uninformative_turns(e["turns"], fl)
            r["n_uninf_turns"] = len(spans)
            records.append(r)
            meta.append({"uninf_spans": spans})
        trainer._ad_meta = meta
        log = os.environ.get("AP_ROLLOUT_LOG")
        if log:
            step = getattr(getattr(trainer, "state", None), "global_step", None)
            with open(log, "a") as f:
                for r in records:
                    f.write(json.dumps({"step": step, **r}, default=str) + "\n")
        return {"prompt_ids": [e["prompt_ids"] for e in eps], "completion_ids": [e["ids"] for e in eps],
                "logprobs": [e["lps"] for e in eps], "env_mask": [e["mask"] for e in eps], "ap_episodes": records,
                "tool_call_counts": [len(e["env"].trace) for e in eps]}

    return rollout_func
