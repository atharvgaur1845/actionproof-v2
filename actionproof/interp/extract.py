"""Residual-stream extraction for probes and SAE features (spec §14).

Probe positions (both are "the token right before the model's next decision"):
  pos-A  after the tool-result turn of each ACKNOWLEDGED write (obs status success/accepted).
         label = 1 if the acknowledged effect did NOT land (F1 phantom, F2 stale, F3 partial,
         F8 ambiguous-without-commit), else 0. F1 is the built-in negative control: its ack is
         byte-identical to a real one, so no internal signal can exist there.
  pos-B  right before the assistant turn that calls submit_report.
         label = 1 if the task is NOT actually done (G == 0).

One forward pass per episode: the Cohere template has the prefix property, so every position
is an index into the full conversation's token sequence.

    python -m actionproof.interp.extract --episodes results/curriculum/rft_tiny_train.jsonl \
        --model runs/rft_tiny_curated/merged --out data/interp/rft_tiny_train.npz
"""
from __future__ import annotations

import argparse
import json
from typing import Any

NOT_LANDED = {"F1_phantom", "F2_stale", "F3_partial"}
ACKED = {"success", "accepted"}


def build_examples(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """(prefix length in messages, label, kind, meta) for one episode record."""
    conv, attempts = rec["conversation"], {a["action_idx"]: a for a in rec.get("attempts", [])}
    trace = rec["trace"]
    out, t = [], 0  # t = pointer into the env trace, consumed in call order
    nth: dict[str, int] = {}  # k-th call of each op (the fault schedule is keyed by (tool, k))
    for j, m in enumerate(conv):
        if m["role"] != "assistant" or j + 1 >= len(conv) or conv[j + 1]["role"] != "user":
            continue
        k = conv[j + 1]["content"].count("Tool result for ")
        block = trace[t:t + k]
        t += k
        if any(s["tool"] == "submit_report" for s in block):
            out.append({"prefix": j, "pos": "B", "label": int(rec["verdict"]["G"] == 0),
                        "fault": "+".join(rec.get("faults") or ["benign"]),
                        "reported": rec["verdict"]["reported"], "task_id": rec["task_id"]})
            continue
        for s in block:
            k_op = nth[s["tool"]] = nth.get(s["tool"], -1) + 1
            at = attempts.get(s["idx"])
            if at is None or at["obs_status"] not in ACKED:
                continue
            landed = not (at["fault"] in NOT_LANDED or (at["fault"] == "F8_ambiguous" and not at["committed"]))
            out.append({"prefix": j + 2, "pos": "A", "label": int(not landed), "fault": at["fault"] or "none",
                        "op": at["op"], "k": k_op, "task_id": rec["task_id"]})
    # submit_report as the very last assistant turn (no following tool-result message)
    if conv and conv[-1]["role"] == "assistant" and trace and trace[-1]["tool"] == "submit_report" \
            and not any(e["pos"] == "B" for e in out):
        out.append({"prefix": len(conv) - 1, "pos": "B", "label": int(rec["verdict"]["G"] == 0),
                    "fault": "+".join(rec.get("faults") or ["benign"]),
                    "reported": rec["verdict"]["reported"], "task_id": rec["task_id"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--model", required=True, help="merged checkpoint (or base) whose activations we read")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    recs = [json.loads(l) for f in a.episodes for l in open(f)][: a.max_episodes]

    def n_tokens(msgs) -> int:
        return len(tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False),
                              add_special_tokens=False))

    feats, meta = [], []
    with torch.no_grad():
        for r in recs:
            exs = build_examples(r)
            if not exs:
                continue
            last = max(e["prefix"] for e in exs)
            text = tok.apply_chat_template(r["conversation"][:last], add_generation_prompt=True, tokenize=False)
            ids = tok.encode(text, add_special_tokens=False)[: a.max_tokens]
            hs = model(torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states
            stack = torch.stack([h[0] for h in hs])  # [L+1, T, d]
            for e in exs:
                pos = n_tokens(r["conversation"][: e["prefix"]]) - 1
                if pos >= len(ids):
                    continue
                feats.append(stack[:, pos, :].float().cpu().numpy().astype(np.float16))
                meta.append({**e, "model": a.model})
    X = np.stack(feats)  # [n, L+1, d]
    np.savez_compressed(a.out, X=X, meta=np.array([json.dumps(m) for m in meta]))
    pos_counts = {p: sum(m["pos"] == p for m in meta) for p in "AB"}
    print(json.dumps({"examples": len(meta), "by_pos": pos_counts, "positives": sum(m["label"] for m in meta),
                      "shape": list(X.shape)}))


if __name__ == "__main__":
    main()
