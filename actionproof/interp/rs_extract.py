"""GPU: residual-stream activations at the pre-decision position + next-tool distribution, for every model on the
SAME prefixes (docs/INTERP_PLAN.md). Optional intervention hook (M3): project a direction out of / add it to the
residual stream at one layer over the final tool-result message.

    python -m actionproof.interp.rs_extract --prefixes results/interp/rs/prefixes.jsonl --kinds A1 A2 \
        --model runs/grpo_tiny_v2_outcome_s0/merged --tag outcome_s0 --out results/interp/rs
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

HEADERS = ('[{"tool_name": "', '[\n    {\n        "tool_name": "')
GROUPS = {"verify": ("query_ledger", "lookup_order", "get_refund_status"),
          "recover": ("issue_refund", "reconcile_order"),
          "proceed": ("add_audit_note", "close_ticket", "submit_report")}
TOOLS = [t for g in GROUPS.values() for t in g]


class Runner:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                          device_map="cuda").eval()
        self.layers = self.model.model.layers
        self.hook = None  # (layer, fn(h, span) -> h)

    def ids(self, conv, gen=True):
        return self.tok.encode(self.tok.apply_chat_template(conv, add_generation_prompt=gen, tokenize=False),
                               add_special_tokens=False)

    def _seq_logprob(self, cache, last_logits, cont_ids):
        """Sum log p(cont | cache) and the cache after it (cache is copied, never mutated)."""
        torch = self.torch
        c = copy.deepcopy(cache)
        lp = torch.log_softmax(last_logits.float(), -1)[cont_ids[0]].item()
        if len(cont_ids) > 1:
            out = self.model(torch.tensor([cont_ids], device="cuda"), past_key_values=c, use_cache=True)
            lps = torch.log_softmax(out.logits[0, :-1].float(), -1)
            lp += sum(lps[i, t].item() for i, t in enumerate(cont_ids[1:]))
            return lp, out.past_key_values, out.logits[0, -1]
        out = self.model(torch.tensor([cont_ids], device="cuda"), past_key_values=c, use_cache=True)
        return lp, out.past_key_values, out.logits[0, -1]

    def run(self, conv, want_tools: bool):
        torch = self.torch
        ids = self.ids(conv)
        span0 = len(self.ids(conv[:-1], gen=False))  # tokens before the final tool-result message
        handle = None
        if self.hook is not None:
            layer, fn = self.hook

            def h(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                if hs.shape[1] > 1:  # prefix pass only
                    hs = fn(hs, span0)
                return (hs, *out[1:]) if isinstance(out, tuple) else hs
            handle = self.layers[layer].register_forward_hook(h)
        try:
            with torch.no_grad():
                out = self.model(torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=want_tools)
                hidden = torch.stack([x[0, -1] for x in out.hidden_states]).float().cpu().numpy()
                probs = None
                if want_tools:
                    if handle is not None:
                        handle.remove(); handle = None  # continuation tokens are not intervened on
                    logz, per = [], {}
                    for hdr in HEADERS:
                        hid = self.tok.encode(hdr, add_special_tokens=False)
                        lph, cache_h, logit_h = self._seq_logprob(out.past_key_values, out.logits[0, -1], hid)
                        for t in TOOLS:
                            tid = self.tok.encode(t + '"', add_special_tokens=False)
                            lpt, _, _ = self._seq_logprob(cache_h, logit_h, tid)
                            per.setdefault(t, []).append(lph + lpt)
                    m = max(v for vs in per.values() for v in vs)
                    raw = {t: sum(math.exp(v - m) for v in vs) for t, vs in per.items()}
                    z = sum(raw.values())
                    probs = {t: raw[t] / z for t in TOOLS}
        finally:
            if handle is not None:
                handle.remove()
        return hidden, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--kinds", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tools", action="store_true")
    a = ap.parse_args()
    import numpy as np
    rows = [r for l in open(a.prefixes) if (r := json.loads(l))["kind"] in a.kinds]
    R = Runner(a.model)
    X, P, ids = [], [], []
    for r in rows:
        h, p = R.run(r["conv"], want_tools=not a.no_tools and r["kind"] in ("A1", "A2"))
        X.append(h.astype(np.float16)); P.append(json.dumps(p)); ids.append(r["id"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.tag}__{'_'.join(a.kinds)}.npz", X=np.stack(X), ids=np.array(ids), probs=np.array(P))
    print(a.tag, a.kinds, len(ids), X[0].shape)


if __name__ == "__main__":
    main()
