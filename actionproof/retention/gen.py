"""R1 capability retention: greedy generation for the frozen IFEval and BFCL inputs (PREREGISTRATION_R1.md §4).

Every model sees `data/retention/r1_inputs.jsonl` through its own chat template, greedy (temperature 0),
in one batched vLLM call per generation budget. Adapters are evaluated as merged checkpoints
(`actionproof/train/merge.py`: vLLM LoRA fails on Cohere's tied embeddings).

An input whose rendered prompt plus its generation budget exceeds the 8,192-token context is not
generated; it is written with `over_length: true`. The set is identical for every model (same inputs,
same tokenizer), and the scorer drops it for every model alike (§5).

    python -m actionproof.retention.gen --model runs/rft_proc/merged --tag warm_start
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CTX = 8192
STOPS = ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>")


def render(tok, messages: list[dict]) -> list[int]:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def plan(tok, items: list[dict]) -> tuple[list[tuple[dict, list[int]]], list[dict]]:
    """Split inputs into (item, prompt ids) to generate and the over-length items."""
    keep, over = [], []
    for it in items:
        ids = render(tok, it["messages"])
        if len(ids) + it["max_tokens"] <= CTX:
            keep.append((it, ids))
        else:
            over.append(it)
    return keep, over


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--inputs", default="data/retention/r1_inputs.jsonl")
    ap.add_argument("--out-dir", default="results/r1/gen")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N inputs per benchmark")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm

    items = [json.loads(line) for line in Path(a.inputs).read_text().splitlines()]
    if a.limit:
        seen: dict[str, int] = {}
        items = [i for i in items if (seen.__setitem__(i["bench"], seen.get(i["bench"], 0) + 1) or seen[i["bench"]] <= a.limit)]
    tok = AutoTokenizer.from_pretrained(a.model)
    keep, over = plan(tok, items)
    stop_ids = sorted({tok.convert_tokens_to_ids(s) for s in STOPS})
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=CTX, gpu_memory_utilization=a.gpu_mem, seed=0,
              enable_prefix_caching=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, timing = [], {}
    for budget in sorted({it["max_tokens"] for it, _ in keep}):
        batch = [(it, ids) for it, ids in keep if it["max_tokens"] == budget]
        sp = SamplingParams(temperature=0.0, max_tokens=budget, stop_token_ids=stop_ids, skip_special_tokens=True)
        t0 = time.time()
        res = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids in batch], sp, use_tqdm=False)
        dt = time.time() - t0
        n_out = 0
        for (it, ids), r in zip(batch, res):
            o = r.outputs[0]
            n_out += len(o.token_ids)
            rows.append({"bench": it["bench"], "id": it["id"], "category": it["category"], "text": o.text,
                         "prompt_tokens": len(ids), "output_tokens": len(o.token_ids),
                         "finish_reason": o.finish_reason, "over_length": False})
        timing[str(budget)] = {"n": len(batch), "seconds": round(dt, 2), "output_tokens": n_out}
    rows += [{"bench": it["bench"], "id": it["id"], "category": it["category"], "text": None,
              "prompt_tokens": None, "output_tokens": 0, "finish_reason": None, "over_length": True} for it in over]
    (out / f"{a.tag}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    merged_from = Path(a.model) / "merged_from.json"
    meta = {"tag": a.tag, "model": a.model, "vllm": vllm.__version__, "n": len(rows), "over_length": len(over),
            "timing_batched": timing, "limit": a.limit, "ctx": CTX,
            "merged_from": json.loads(merged_from.read_text()) if merged_from.exists() else None}
    (out / f"{a.tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
