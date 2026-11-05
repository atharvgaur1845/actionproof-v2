"""ROI comparison (hackathon rules: optional but preferred): closed-model API billing vs self-hosting the
fine-tuned Tiny Aya agent, at a realistic monthly ticket volume.

Workload is MEASURED from our agent's own episodes (LLM calls, input/output tokens per ticket; the
system+task prefix is the cacheable part). Self-hosted throughput is MEASURED by a timed batched vLLM
run (slurm/tasks/175_roi_throughput.sh). Prices come from configs/roi_prices.json with source + date.
Quality (VSC) is reported next to cost: a cheap agent that lies is not a saving.

    python -m actionproof.eval.roi --episodes results/eval/r3/grpo_tiny_v2_pcstr_d_s0__neutral__test_iid.jsonl \
        --throughput results/roi/throughput.json --prices configs/roi_prices.json --out results/roi/roi.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def workload(path: str, tokenizer: str = "CohereLabs/tiny-aya-global") -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    enc = lambda s: len(tok.encode(s, add_special_tokens=False))
    eps = [json.loads(l) for l in open(path)]
    calls = inp = out = cached = 0
    for e in eps:
        conv = e["conversation"]
        prefix = enc(tok.apply_chat_template(conv[:2], tokenize=False, add_generation_prompt=True))
        for i, m in enumerate(conv):
            if m["role"] == "assistant":
                calls += 1
                inp += enc(tok.apply_chat_template(conv[:i], tokenize=False, add_generation_prompt=True))
                out += enc(m["content"])
                cached += prefix if i > 2 else 0  # the prefix is re-sent on every later call
    n = len(eps)
    return {"episodes": n, "vsc": sum(e["vsc"] for e in eps) / n, "calls": calls / n, "input_tok": inp / n,
            "cacheable_input_tok": cached / n, "output_tok": out / n}


def table(w: dict, thr: dict, prices: dict) -> str:
    vol = prices["monthly_tickets"]
    rows = [f"Workload per ticket (measured, n={w['episodes']}): {w['calls']:.1f} LLM calls, "
            f"{w['input_tok']:,.0f} input tokens ({w['cacheable_input_tok']:,.0f} cacheable prefix), "
            f"{w['output_tok']:,.0f} output tokens. Self-hosted agent VSC = {w['vsc']:.2f}.",
            f"Self-hosted throughput (measured on {thr['gpu']}): {thr['episodes_per_hour']:,.0f} tickets/GPU-hour "
            f"({thr['episodes']} tickets in {thr['seconds']:.0f} s, batched vLLM).", "",
            f"| Option | $/1k tickets | $/month at {vol:,} tickets | Notes |", "|---|---|---|---|"]
    for g in prices["gpu"]:
        if g.get("usd_per_hour") is None:
            continue
        per = g["usd_per_hour"] / thr["episodes_per_hour"]  # assumes the measured GPU class; scale for others
        rows.append(f"| Self-hosted Tiny Aya, {g['name']} | {1000 * per:,.3f} | {vol * per:,.2f} | "
                    f"{vol / thr['episodes_per_hour']:,.1f} GPU-h/month; price: {g['source']} ({g['date']}) |")
    for c in prices["closed_api"]:
        if c.get("usd_per_mtok_input") is None or c.get("usd_per_mtok_output") is None:
            continue
        cin = c.get("usd_per_mtok_cached_input")
        unc = (w["input_tok"] - (w["cacheable_input_tok"] if cin is not None else 0)) * c["usd_per_mtok_input"]
        cac = w["cacheable_input_tok"] * cin if cin is not None else 0.0
        per = (unc + cac + w["output_tok"] * c["usd_per_mtok_output"]) / 1e6
        rows.append(f"| {c['name']} API | {1000 * per:,.3f} | {vol * per:,.2f} | "
                    f"{'with' if cin is not None else 'without'} prompt caching; {c['source']} ({c['date']}) |")
    rows.append("\nGPU rows other than the measured class are priced at the measured throughput (an upper bound "
                "for slower GPUs); the measured row is exact.")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--throughput", required=True)
    ap.add_argument("--prices", default="configs/roi_prices.json")
    ap.add_argument("--out", default="results/roi/roi.md")
    a = ap.parse_args()
    md = table(workload(a.episodes), json.loads(Path(a.throughput).read_text()), json.loads(Path(a.prices).read_text()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
