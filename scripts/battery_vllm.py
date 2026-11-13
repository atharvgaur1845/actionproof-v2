"""Forced-prefix battery, corrected, plus the greedy action on each prefix (task 416a; SPEC §27 2026-11-13).

Why: the registered battery (`actionproof/eval/forced_prefix.py`, hashed, left unchanged) scores each candidate as
`stem + name + '"'`. In real outputs the name is followed by `", "parameters"`, and the tokenizer emits `",` as ONE
token, so the battery's final token (a lone `"`) is one the model almost never produces there. Its probability
multiplies into every candidate's score by an amount that depends on the preceding name, so the renormalised
readout is biased. Check: on the 50 credit prefixes (type e) the registered battery's argmax matched the source
model's own greedy action in 2/50.

Two modes, one process each, both on the MERGED bf16 weights the evaluation served:
  --mode greedy  vLLM `llm.chat(prefix)` at temperature 0 (as `run_eval.run`); first tool name and report status.
  --mode score   exact log P(stem + candidate + '",') with HF, renormalised over the registered candidate sets.
                 The shared part (prefix + stem + common candidate tokens) is run once per readout and its KV cache
                 reused for each candidate's 1-6 tail tokens; the first rows are checked against a no-cache pass.

    python scripts/battery_vllm.py --mode greedy --prefixes data/bench/a7/forced_prefixes.jsonl --tag m \
        --weights runs/grpo_tiny_a6_pcstr_d_s0/merged --out results/a7/battery_v2_greedy.jsonl
    python scripts/battery_vllm.py --mode score  ... --out results/a7/battery_v2_scores.jsonl
    python scripts/battery_vllm.py --mode score --dry ...   # CPU: build and check the requests only
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actionproof.eval.forced_prefix import CREDIT_TOOL, REPORT_STEM, STATUSES, TOOL_STEM, TOOLS  # noqa: E402

CLOSE = '",'
TOOL_RE = re.compile(r'"tool_name":\s*"([A-Za-z_]+)"')
STATUS_RE = re.compile(r'"submit_report",\s*"parameters":\s*\{\s*"status":\s*"([A-Za-z_]+)"')
N_CHECK = 3


def renorm(lps: list[float]) -> list[float]:
    m = max(lps)
    e = [math.exp(x - m) for x in lps]
    return [x / sum(e) for x in e]


def groups(r: dict):
    if "status" in r["readouts"]:
        yield "status", REPORT_STEM, list(STATUSES)
    if "tool" in r["readouts"]:
        yield "tool", TOOL_STEM, list(TOOLS) + ([CREDIT_TOOL] if r["schema"] == "a4" else [])


def build(tok, r: dict):
    """Per readout: (key, candidates, token sequences, length of the shared token prefix)."""
    text = tok.apply_chat_template(r["conversation"], add_generation_prompt=True, tokenize=False)
    close_id = tok(CLOSE, add_special_tokens=False).input_ids
    assert len(close_id) == 1, close_id
    for key, stem, cands in groups(r):
        seqs = [tok(text + stem + c + CLOSE, add_special_tokens=False).input_ids for c in cands]
        assert all(s[-1] == close_id[0] for s in seqs), (r["id"], key)
        lcp = 0
        while all(len(s) > lcp for s in seqs) and len({s[lcp] for s in seqs}) == 1:
            lcp += 1
        assert lcp > 0 and all(len(s) > lcp for s in seqs), (r["id"], key, lcp)
        yield key, cands, seqs, lcp


def greedy(a, rows) -> None:
    from vllm import LLM, SamplingParams
    llm = LLM(a.weights, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16", seed=0)
    outs = llm.chat([r["conversation"] for r in rows], SamplingParams(temperature=0.0, max_tokens=64, seed=0),
                    use_tqdm=False)
    with open(a.out, "a") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            mt, ms = TOOL_RE.search(text), STATUS_RE.search(text)
            f.write(json.dumps({"model": a.tag, "id": r["id"], "type": r["type"], "schema": r["schema"],
                                "greedy_text": text[:160], "greedy_tool": mt.group(1) if mt else None,
                                "greedy_status": ms.group(1) if ms else None}) + "\n")


def score(a, rows) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.weights)
    if a.dry:
        n = [(len(s), len(s) - lcp) for r in rows for _, _, seqs, lcp in build(tok, r) for s in seqs]
        print(json.dumps({"rows": len(rows), "candidates": len(n), "max_len": max(x for x, _ in n),
                          "max_tail": max(t for _, t in n), "mean_tail": sum(t for _, t in n) / len(n)}))
        return
    model = AutoModelForCausalLM.from_pretrained(a.weights, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def full(s: list[int], lcp: int) -> float:  # reference: no cache
        lp = torch.log_softmax(model(torch.tensor([s]).cuda()).logits[0].float(), -1)
        return sum(lp[p - 1, s[p]].item() for p in range(lcp, len(s)))

    @torch.no_grad()
    def cached(seqs: list[list[int]], lcp: int) -> list[float]:
        out = model(torch.tensor([seqs[0][:lcp]]).cuda(), use_cache=True)
        first = torch.log_softmax(out.logits[0, -1].float(), -1)
        tot = []
        for s in seqs:
            v = first[s[lcp]].item()
            if len(s) > lcp + 1:
                o = model(torch.tensor([s[lcp:-1]]).cuda(), past_key_values=copy.deepcopy(out.past_key_values),
                          use_cache=True)
                lp = torch.log_softmax(o.logits[0].float(), -1)
                v += sum(lp[j, s[lcp + 1 + j]].item() for j in range(len(s) - lcp - 1))
            tot.append(v)
        return tot

    use_full = False
    with open(a.out, "a") as f:
        for i, r in enumerate(rows):
            rec = {"model": a.tag, "id": r["id"], "type": r["type"], "schema": r["schema"], "close": CLOSE}
            for key, cands, seqs, lcp in build(tok, r):
                lps = [full(s, lcp) for s in seqs] if use_full else cached(seqs, lcp)
                if i < N_CHECK and not use_full:  # the KV-cache path must reproduce the no-cache log-probability
                    ref = [full(s, lcp) for s in seqs]
                    if max(abs(x - y) for x, y in zip(lps, ref)) > 0.05:
                        print("KV-cache mismatch, falling back to full passes", r["id"], key, lps, ref, file=sys.stderr)
                        use_full, lps = True, ref
                rec["scorer"] = "full" if use_full else "kv_cache"
                rec[key] = dict(zip(cands, renorm(lps)))
                rec[key + "_logp"] = dict(zip(cands, lps))
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("greedy", "score"), required=True)
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True, help="merged checkpoint directory (or base model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prefixes)]
    (greedy if a.mode == "greedy" else score)(a, rows)
    print("battery_v2", a.mode, a.tag, len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
