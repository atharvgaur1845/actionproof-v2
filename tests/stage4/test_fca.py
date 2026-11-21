"""FCA checks (docs/research/algorithm/FCA_GRPO.md): advantage arithmetic, forced-action isolation from hidden state,
and an end-to-end CPU run of the FCA rollout with a stub tokenizer and stub vLLM (masks, spans, forced log-probs)."""
import json
import sys
import types

import pytest

pytest.importorskip("transformers")

from actionproof.bench.generator_proc import DATA, load_tasks  # noqa: E402
from actionproof.stage3.env import make_env_s3  # noqa: E402
from actionproof.stage3.rollout import start_conversation  # noqa: E402
from actionproof.stage4 import forks  # noqa: E402
from actionproof.stage4.advantage import fca_advantages  # noqa: E402

DEV = [t for t in load_tasks(DATA / "s3_rec_dev.jsonl") if t.meta.get("start_type") in ("ack", "read")]


def test_advantage_arithmetic():
    meta = [{"group": "g", "kind": "on", "cls": "proceed", "span": None}] * 2 + \
           [{"group": "g", "kind": "forced", "cls": "compensate", "span": (1, 3)},
            {"group": "g", "kind": "forced", "cls": "proceed", "span": (0, 2)},
            {"group": "h", "kind": "plain", "cls": None, "span": None}]
    rewards = [0.0, 0.0, 1.0, 0.0, 1.0]
    adv, diag = fca_advantages(rewards, meta, 4, base=[9, 9, 9, 9, 0.5])
    sigma = diag[0]["sigma"]
    assert diag[0]["V"] == 0.0 and diag[0]["on_zero_var"]          # all on-policy rollouts failed identically
    assert adv[0] == [0.0] * 4 and adv[1] == [0.0] * 4
    assert adv[2] == [0.0, pytest.approx(1 / sigma), pytest.approx(1 / sigma), 0.0]   # credit only on the forced span
    assert adv[3] == [0.0, 0.0, 0.0, 0.0]                         # proceed: Q = 0 = V
    assert adv[4] == [0.5] * 4                                    # plain groups keep TRL's advantage


def test_forced_actions_use_visible_information_only():
    for t in DEV[:40]:
        env = make_env_s3(t, "x")
        conv = start_conversation(env, t, "")
        hidden = type(t)(t.task_id, t.split, t.workflow, t.prompt, {k: [] for k in t.initial_state},
                         {**t.meta, "faults": [], "never_commit": None})   # hidden state removed
        for c in forks.available(t, conv):
            assert forks.forced_action(t, conv, c) == forks.forced_action(hidden, conv, c), (t.task_id, c)
        req = forks.request(t.prompt)
        po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
        assert req == {"po_id": po["po_id"], "qty": po["qty"], "amount_cents": po["amount_cents"]}


def test_compensate_is_the_repair_on_partial_payment():
    hits = 0
    for t in DEV:
        f = (t.meta.get("faults") or [{}])[0]
        if f.get("kind") == "partial" and t.meta["start_step"] == "pay" and t.meta["start_type"] == "read":
            env = make_env_s3(t, "x")
            conv = start_conversation(env, t, "")
            tool, args = forks.forced_action(t, conv, "compensate")
            live = [p for p in env.state["payments"] if p["po_id"] == t.meta["po_id"] and p["kind"] == "payment"]
            assert tool == "refund_payment" and args["payment_id"] == live[-1]["payment_id"]
            hits += 1
    assert hits > 0


class _Tok:
    """Character-level stub tokenizer with a prefix-preserving chat template."""
    def apply_chat_template(self, conv, add_generation_prompt=False, tokenize=False):
        s = "".join(f"<{m['role']}>{m['content']}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>" if m["role"] == "assistant"
                    else f"<{m['role']}>{m['content']}" for m in conv)
        return s + ("<assistant>" if add_generation_prompt else "")

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<|END_OF_TURN_TOKEN|>": 1, "<|END_RESPONSE|>": 2}[t]


def test_rollout_masks_spans_and_forced_logprobs(monkeypatch):
    from actionproof.stage4 import rollout_fca
    report = json.dumps([{"tool_name": "submit_report", "parameters": {"status": "completed", "summary": "x"}}])

    class _Out:
        def __init__(self):
            self.outputs = [types.SimpleNamespace(token_ids=[ord(c) for c in report],
                                                  logprobs=[{ord(c): types.SimpleNamespace(logprob=-0.5)} for c in report])]

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **k: k
    vinputs = types.ModuleType("vllm.inputs")
    vinputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", vinputs)
    task = next(t for t in DEV if t.meta["start_type"] == "read")
    tr = types.SimpleNamespace(processing_class=_Tok(), state=types.SimpleNamespace(global_step=0),
                               vllm_generation=types.SimpleNamespace(llm=types.SimpleNamespace(
                                   generate=lambda ps, sampling_params, use_tqdm: [_Out() for _ in ps])))
    rf = rollout_fca.make_rollout_func([task], "", n_on=4)
    out = rf([[{"role": "user", "content": task.prompt}]] * 8, tr)
    meta = tr._fca_meta
    assert [m["kind"] for m in meta] == ["on"] * 4 + ["forced"] * 4
    prompts = {tuple(p) for p in out["prompt_ids"]}
    assert len(prompts) == 1                                              # same visible decision state for all branches
    for m, ids, mask, lps in zip(meta, out["completion_ids"], out["env_mask"], out["logprobs"]):
        assert len(ids) == len(mask) == len(lps)
        if m["kind"] == "forced":
            s, e = m["span"]
            assert s == 0 and all(mask[s:e]) and all(x == 0.0 for x in lps[s:e])   # forced: trainable, mu = 1
            text = "".join(chr(i) for i in ids[s:e])
            tool, args = forks.forced_action(task, start_conversation(make_env_s3(task, "y", seed=0), task, ""), m["cls"])  # rollouts use seed 0 (see CONSISTENCY_AUDIT)
            assert json.loads(text) == [{"tool_name": tool, "parameters": args}]
        else:
            assert any(x == -0.5 for x in lps)                           # sampled tokens keep vLLM log-probs
