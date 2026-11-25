# B1 addendum 1: open frontier models through the HF router (descriptive)

Written 2026-11-26, about 01:00 IST, before any API episode exists. It is the addendum foreseen in
`PREREGISTRATION_B1.md`. B1's design is unchanged otherwise.

**Models.** Router ids are pinned to one provider; the prices are those recorded for J1:

| Tag | Model and provider | Price ($ per 1M input / output) | Max new tokens |
|---|---|---|---|
| qwen | `Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra` | 0.09 / 0.55 | 256 |
| gptoss | `openai/gpt-oss-120b:baseten` (reasoning effort "low") | 0.10 / 0.50 | 1,024 |

**Client.** `scripts/b1_api_eval.py`. It drives `scripts/p1_eval.py::run` unchanged:
- the same system prompt, tool format, tool results, 16-turn cap and verifier;
- greedy (temperature 0), one run per task;
- P1's 8,192-token context guard, applied with an approximate tokenizer (3.5 characters per token).

**Stated asymmetry.** gpt-oss is a reasoning model. Its reasoning tokens count against the new-token limit, so it
gets 1,024 new tokens instead of 256.

**Budget.** A hard stop at $10 cumulative across `results/b1/api_cost.jsonl`. The cost is token-priced from each
response's `usage`. It is logged in `docs/stage2/HF_CREDITS.md` when the runs finish.

**Order.** qwen on all five P1 test splits first, then gptoss, until the cap. Any split not run is reported as not
run.

**Outputs.** The same metrics as B1. Greedy only, so they are compared with the local models' greedy numbers.
Nothing is called supported.

## Deviation D-B1-1 (2026-11-26, about 01:45 IST)

The `aya32b` run with fp8 weights failed. On the A100, vLLM had no native FP8 support and fell back to Marlin
weight-only FP8 kernels. All 600 episodes on `p1_test_full` emitted a degenerate string of "(" characters, ending
with no tool call on turn 1; the other splits behaved the same way.

This is an infrastructure failure, not a model result. The files are moved, not deleted, to
`results/b1/failed_fp8/` and are excluded from every analysis.

The model is rerun once, unquantised in bf16 (`--gpu-mem 0.95`, which fits on one 80 GB GPU), under the tag
`aya32b_bf16`. Nothing else changes.
