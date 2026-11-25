# B1: prompted, untrained models on the frozen P1 test (descriptive)

Registered 2026-11-26, about 00:00 IST, before any B1 output exists. Governing plan:
`docs/strategy/COMPETITIVE_ASSESSMENT_V2.md`.

## Question

How do untrained models, up to ten times the size of the trained 3.35B agent, handle misleading acknowledgements on
identical tasks? This measures the problem's importance (how often capable models believe the acknowledgement) and
places the trained agent D against larger models.

## Models (core family: Cohere Labs; supplementary: open frontier)

| Tag | Model | Decoding | Engine |
|---|---|---|---|
| tag_base | CohereLabs/tiny-aya-global (no adapter) | 1 greedy + 4 sampled (T = 1) | vLLM bf16 |
| aya8b | CohereLabs/aya-expanse-8b | 1 greedy + 4 sampled | vLLM bf16 |
| aya32b | CohereLabs/aya-expanse-32b | 1 greedy + 4 sampled | vLLM, fp8 weights (one 80 GB GPU) |
| qwen235b, gptoss120b (supplementary) | Qwen3-235B, gpt-oss-120b through HF Inference Providers | greedy, 1 run | API; hard stop $10 in code, logged in `docs/stage2/HF_CREDITS.md` |

The API runs are added in a dated addendum before they start, naming the exact provider model ids and the
client script.

## Fixed design

- **Tasks:** the frozen P1 test, all five splits (450 tasks), unchanged.
- **Episode loop:** `scripts/p1_eval.py` `run` and `summarise`, imported unchanged by `scripts/b1_eval.py`. Same
  system prompt, tool format, 16 turns, 256 new tokens, context guard and verifier.
- **Comparators** (already on disk): W, A, Dw, D, Dn and the oracle.

## Outputs (all descriptive; no hypothesis test)

- Per model and split: sampled and greedy VSC, false success, duplicate unrefunded, partial-pay repair, read-back
  after a phantom "ok", calls per success, wall-clock seconds.
- Paired task differences against D, with a 10,000-resample task bootstrap; reported as exploratory.
- **Fp8 caveat:** fp8 weights may change the 32B model's behaviour slightly; this is stated wherever it is reported.

Nothing in B1 is called "supported". No P1, P2 or P3 file is modified.
