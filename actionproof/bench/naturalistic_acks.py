"""Naturalistic ambiguous acknowledgements for F8 (spec §9, upgrade 4). Generated once, then frozen.

Pipeline: Aya Expanse 32B writes one-line service acknowledgements that neither confirm nor deny that
the write took effect -> lexical gate (no explicit outcome words) -> CuratorKIT exact + MinHash dedup
-> ambiguity gate (the same 32B labels each text CONFIRMS / DENIES / UNCLEAR; keep UNCLEAR) ->
actionproof/bench/data/f8_natural.json with provenance. Used by the `test_f8_natural` split.

    python -m actionproof.bench.naturalistic_acks --out actionproof/bench/data/f8_natural.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CONTEXTS = {
    "refund": "a payments API that was asked to refund a customer's order",
    "cancel": "an order-management API that was asked to cancel an order",
    "reversal": "a core-banking API that was asked to reverse a disputed card transaction",
}
GEN_PROMPT = """You are writing realistic test data for a software system.

Write ONE status message, as {system} might return it in the "detail" field of its response. The
message must be AMBIGUOUS: it must neither confirm that the action was carried out nor say that it
failed. Real systems say things like "queued", "received", "pending review", "forwarded to the
processor", "will be reflected shortly". Vary wording, tone and length (6 to 25 words). No ids,
amounts or placeholders. Output only the message text."""
CHECK_PROMPT = """A software agent sent a write request and the service replied with this message:

"{text}"

Does the message state that the action has been COMPLETED, state that it FAILED, or leave it UNCLEAR
whether the action took effect? Answer with exactly one word: CONFIRMS, DENIES or UNCLEAR."""
EXPLICIT = re.compile(r"\b(complete[ds]?|completion|success\w*|succeed\w*|fail\w*|error\w*|declin\w*|"
                      r"reject\w*|done|finished|refunded|cancell?ed|reversed|confirmed)\b", re.I)


def lexical_ok(t: str) -> bool:
    n = len(t.split())
    return 6 <= n <= 25 and not EXPLICIT.search(t) and not re.search(r"[\d{}<>\[\]]", t)


def main() -> None:
    from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
    from curatorkit.schema import DataSample
    from vllm import LLM, SamplingParams
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/aya-expanse-32b")
    ap.add_argument("--per-context", type=int, default=80)
    ap.add_argument("--out", default="actionproof/bench/data/f8_natural.json")
    a = ap.parse_args()
    llm = LLM(model=a.model, max_model_len=2048, gpu_memory_utilization=0.9, quantization="fp8", seed=0)
    out, stats = {}, {}
    for i, (ctx, system) in enumerate(CONTEXTS.items()):
        # one seed per request: a single shared seed makes every sample identical (cf. Amendment 3)
        gens = llm.chat([[{"role": "user", "content": GEN_PROMPT.format(system=system)}]] * a.per_context,
                        [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=60, seed=1000 * i + k)
                         for k in range(a.per_context)])
        texts = [g.outputs[0].text.strip().strip('"').strip() for g in gens]
        lex = [t for t in texts if lexical_ok(t)]
        samples = [DataSample(source_uri=f"aya-expanse-32b/{ctx}", output=t, task_type="language_modeling") for t in lex]
        samples = MinHashDeduplicator(threshold=0.7).run(ExactDeduplicator().run(samples))
        kept = [s.output for s in samples]
        checks = llm.chat([[{"role": "user", "content": CHECK_PROMPT.format(text=t)}] for t in kept],
                          SamplingParams(temperature=0.0, max_tokens=4))
        labels = [c.outputs[0].text.strip().upper() for c in checks]
        final = [t for t, l in zip(kept, labels) if l.startswith("UNCLEAR")]
        out[ctx] = final
        stats[ctx] = {"generated": len(texts), "lexical": len(lex), "dedup": len(kept), "unclear": len(final)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"provenance": {"model": a.model, "gen_prompt": GEN_PROMPT, "check_prompt": CHECK_PROMPT,
                                                      "gates": ["lexical", "curatorkit.exact", "curatorkit.minhash@0.7",
                                                                "aya32b ambiguity=UNCLEAR"], "stats": stats},
                                       "texts": out}, indent=1, ensure_ascii=False))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
