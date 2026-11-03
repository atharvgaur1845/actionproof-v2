"""Batched multi-turn GRPO rollouts on TRL's colocated vLLM engine (spec §13.3).

AgentTune's Aya tool loop runs rollouts one at a time (G0: 415 s for 8 rollouts). This
engine advances ALL of a step's rollouts in lockstep: one batched vLLM call per turn.

Token-level design (no retokenisation drift):
  * the policy sees `prompt_ids + completion_ids` as raw token ids (TokensPrompt), so the
    tokens it sampled are byte-for-byte the tokens it is trained on;
  * sampled tokens get env_mask=1 and their vLLM logprob; template/tool-result tokens get
    env_mask=0 and logprob 0.0 (no loss), exactly AgentTune's convention;
  * tool-result segments are the chat-template delta between "conversation + assistant" and
    "conversation + assistant + user(tool results)" — valid because the Cohere template has
    the prefix property in both directions (verified on Tiny Aya, see tests/notes).
Prompts, tool schemas, the parser and the tool-result text are AgentTune's / ours exactly
as in evaluation, so the policy is trained in the format it is evaluated in.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

from actionproof.bench.generator import Task
from actionproof.bench.prompts import TOOL_RESULT_FORMAT
from actionproof.rollout.agenttune_rollout import episode_record, make_env, prompt_key
from actionproof.rollout.episode import _agenttune, system_prompt

ASSISTANT_TAIL = "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"


def _render(tok, conv, gen: bool) -> str:
    return tok.apply_chat_template(conv, add_generation_prompt=gen, tokenize=False)


def make_batched_rollout_func(tasks: list[Task], base_prompt: str, max_turns: int = 10,
                              max_tokens: int = 256, temperature: float = 1.0,
                              max_context: int = 7168) -> Callable:
    by_prompt = {t.prompt: t for t in tasks}
    _, _, parse = _agenttune()

    def rollout_func(prompts: list[Any], trainer: Any = None, **_: Any) -> dict[str, Any]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        tok = trainer.processing_class
        llm = trainer.vllm_generation.llm
        tail_ids = tok.encode(ASSISTANT_TAIL, add_special_tokens=False)
        stop_ids = sorted({tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>"),
                           tok.convert_tokens_to_ids("<|END_RESPONSE|>")})
        eps = []
        for p in prompts:
            task = by_prompt[prompt_key(p)]
            rid = f"ro-{uuid.uuid4().hex[:12]}"
            env = make_env(task, rid, seed=uuid.UUID(rid[3:].ljust(32, "0")).int % 2**31)
            conv = [{"role": "system", "content": system_prompt(env, base_prompt)},
                    {"role": "user", "content": task.prompt}]
            pids = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
            if isinstance(pids, dict):
                pids = pids["input_ids"]
            eps.append({"task": task, "env": env, "conv": conv, "prompt_ids": list(pids),
                        "ids": [], "mask": [], "lps": [], "done": False})

        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=0,
                            stop_token_ids=stop_ids)
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
                # sampled tokens are trainable; then close the turn exactly as the template does
                while gen and gen[-1] in stop_ids:  # drop sampled stop tokens; the tail re-adds them
                    gen.pop(); lps.pop()
                text = tok.decode(gen, skip_special_tokens=True)
                e["ids"] += gen + tail_ids
                e["mask"] += [1] * len(gen) + [0] * len(tail_ids)
                e["lps"] += lps + [0.0] * len(tail_ids)
                before = _render(tok, e["conv"] + [{"role": "assistant", "content": text}], gen=False)
                e["conv"].append({"role": "assistant", "content": text})
                calls = parse(text) or []
                if not calls:
                    e["done"] = True
                    continue
                results = []
                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = None
                    obs = e["env"].call(name, args) if isinstance(args, dict) else \
                        {"status": "error", "error": "invalid_arguments"}
                    results.append(TOOL_RESULT_FORMAT.format(name=name, content=json.dumps(obs)))
                e["conv"].append({"role": "user", "content": "\n\n".join(results)})
                if e["env"].done:
                    e["done"] = True
                    continue
                after = _render(tok, e["conv"], gen=True)
                if not after.startswith(before):
                    raise RuntimeError("chat template lost the prefix property; token continuation unsafe")
                obs_ids = tok.encode(after[len(before):], add_special_tokens=False)
                if len(e["prompt_ids"]) + len(e["ids"]) + len(obs_ids) + max_tokens > max_context:
                    e["done"] = True  # out of context: end the episode (no report -> scored as such)
                    continue
                e["ids"] += obs_ids
                e["mask"] += [0] * len(obs_ids)
                e["lps"] += [0.0] * len(obs_ids)

        records = [episode_record(e["task"], e["env"]) for e in eps]
        ids = [r["rollout_id"] for r in records]
        assert len(set(ids)) == len(prompts), "duplicate rollout ids"
        assert all(r["receipt_rollout_ids"] in ([], [r["rollout_id"]]) for r in records)
        return {
            "prompt_ids": [e["prompt_ids"] for e in eps],
            "completion_ids": [e["ids"] for e in eps],
            "logprobs": [e["lps"] for e in eps],
            "env_mask": [e["mask"] for e in eps],
            "ap_episodes": records,
            "tool_call_counts": [sum(1 for s in e["env"].trace) for e in eps],
        }

    return rollout_func
