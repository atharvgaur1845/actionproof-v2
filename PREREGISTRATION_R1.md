# PREREGISTRATION R1: general-capability retention after ProcureSim training (IFEval, BFCL)

Registered 2026-11-23, before any R1 generation was run on any model. The registration time is in `results/r1_hashes.json`, written by `scripts/freeze_r1.py`. Branch `actionproof-research-upgrade`.

## 1. Question

Does ProcureSim training (SFT warm start, then GRPO) damage what Tiny Aya Global could already do outside ProcureSim? The tests are:
- **function calling** on unseen schemas (BFCL);
- **general instruction following** (IFEval).

The selection reasoning, including why τ-bench is secondary and AgentBench is excluded, is in `docs/upgrade/PLAN.md` §2. τ-bench gets its own addendum, registered before any τ-bench output.

## 2. Models (tag = training run)

The procurement chain:
- `base` = CohereLabs/tiny-aya-global;
- `warm_start` = `runs/rft_proc` merged;
- every adapter below is merged onto `runs/rft_proc/merged`.

| Arm | Tags | Runs |
|---|---|---|
| A12 state reward (R_V) | `a12_rv_s0`–`s4` | `runs/a12_proc_rv_s*` |
| A12 acknowledgement reward (R_A) | `a12_ra_s0`–`s4` | `runs/a12_proc_ra_s*` |
| Recovery-start curriculum (RSC) | `rsc_s0`–`s4` | `runs/s3_rv_mix_s*` |
| Class-matched control | `control_s0`–`s4` | `runs/s3_matched_s*` |
| Acknowledgement-only recovery starts | `ackonly_s0`–`s4` | `runs/s3_ackonly_s*` |
| Tool-name randomisation (SR1) | `sr1_s0`–`s2` | `runs/s3_sr_mix_s*` |

The Stage 1 chain is exploratory. `s1_warm_start` = `runs/rft_tiny_v2` merged; the adapters below are merged onto it.
- `s1_pcstr_d_s0`, `s1_pcstr_d_s1` (`runs/grpo_tiny_v2_pcstr_d_s*`);
- `s1_tool_trust_s0`, `s1_tool_trust_s1` (`runs/grpo_tiny_v2_tool_trust_s*`).

Stage 5 or later candidates are added by an addendum, with the same pipeline and the same analysis.

## 3. Benchmarks and frozen inputs

`data/retention/r1_inputs.jsonl` holds 4,032 inputs. It is built by `actionproof/retention/build_prompts.py` in the scoring environment `configs/r1_scoring_requirements.txt`.

**IFEval** (google/IFEval, 541 prompts, Apache-2.0):
- the single user turn, as in lm-eval 0.4.13;
- 1,280 new tokens.

**BFCL v4** (bfcl-eval 2026.3.23, Apache-2.0), the 11 single-turn Python categories: simple_python 400, multiple 200, parallel 200, parallel_multiple 200, irrelevance 240, live_simple 258, live_multiple 1,053, live_parallel 16, live_parallel_multiple 24, live_irrelevance 884, live_relevance 16.
- Prompting mode, with BFCL's own default system prompt (`system_prompt_pre_processing_chat_model`: python call format, json docs, plaintext, classic).
- 512 new tokens.

**Excluded, with reasons fixed now:**
- BFCL multi-turn, memory and web search: they need BFCL's executable backends and multi-step harness, and τ-bench covers stateful multi-turn.
- Format sensitivity: a prompt-variation study, not a capability measure.
- Java and JavaScript: not the setting.

**Independence:** ProcureSim training data contains none of these prompts, schemas or formatting constraints. The benchmarks could be in Tiny Aya's own pre- or post-training data. That would shift absolute scores, but every compared model shares the base, so the differences are still interpretable.

## 4. Generation

`actionproof/retention/gen.py`:
- vLLM, bf16, greedy (temperature 0), seed 0;
- context 8,192;
- each model's own chat template;
- adapters evaluated as merged checkpoints (`actionproof/train/merge.py`);
- each model generated once.

**Over-length rule:** an input whose rendered prompt plus its token budget exceeds 8,192 is not generated. That set is identical for every model, is dropped for every model alike, and its size is reported.

A smoke test (`--limit 20`, base model only) checks the pipeline. Its outputs are deleted and not analysed.

## 5. Scoring

`scripts/r1_score.py`, the official checkers:
- **IFEval:** lm-eval `process_results`, prompt- and instruction-level, strict and loose.
- **BFCL:** the `eval_runner` logic.
  - AST decoding, the format check and `ast_checker` for the call categories; decoding failures are also counted as a format-failure rate.
  - No decodable call = correct for irrelevance.
  - A decodable call = correct for live_relevance.

**Endpoints:**
- `ifeval_prompt_strict`;
- `bfcl_nonlive_ast`: unweighted mean of the four non-live AST categories, BFCL's aggregation;
- `bfcl_irrelevance`;
- `bfcl_live_ast`: live AST categories, category means averaged;
- `bfcl_live_irrelevance`.

## 6. Analysis (`scripts/r1_stats.py`, frozen with this file)

For an arm with seeds 1..S: Δ = E(seed-averaged per-item scores) − E(warm-start per-item scores). E is the per-category accuracy averaged over the endpoint's categories.

Intervals use a paired bootstrap over items, stratified by category:
- B = 10,000, seed 20261123;
- trained models held fixed, so the interval reflects task-level uncertainty only;
- per-seed Δs are reported next to it.

| ID | Hypothesis | Test | Supported if |
|---|---|---|---|
| **H1** | The SFT warm start loses capability against the base | `warm_vs_base` on `ifeval_prompt_strict` and `bfcl_nonlive_ast` | the 95% CI upper bound is below 0, per endpoint |
| **H2 (primary)** | GRPO does not lose more than 3 points beyond its warm start | `{a12_rv, rsc}_vs_warm` on `ifeval_prompt_strict` and `bfcl_nonlive_ast`. Bootstrap p = P(Δ* ≤ −0.03); Holm over the 4 tests | Holm p < 0.05 for that test (non-inferior) |
| **H3** | ProcureSim training, which always rewards acting through tools, lowers abstention when no tool fits | `{a12_rv, rsc}_vs_warm` on `bfcl_irrelevance`. Bootstrap p = P(Δ* ≥ 0); Holm over 2 | Holm p < 0.05 |
| H4 (exploratory) | Tool-name randomisation helps on unseen schemas | `sr1` (3 seeds) − `rsc` (5 seeds) on `bfcl_live_ast` and `bfcl_nonlive_ast` | descriptive only |

**Descriptive, not tested:**
- every other arm and endpoint;
- the format-failure rate;
- output tokens;
- batched throughput.

**Not claimed:**
- seed-level significance: five runs per arm;
- comparability with the public BFCL leaderboard: prompting mode, an unsupported model handler, and our own generation loop.

## 7. Conditional mitigation (exploratory, registered now)

If any H2 test fails, one mitigation is tested with no tuning: merging the GRPO adapter at LoRA scale 0.5. It is evaluated on the same R1 inputs and on ProcureSim `proc_dev`, and labelled exploratory. It is not used to revise any H1–H3 verdict.

## 8. Deviations

Every deviation is logged in `docs/stage2/DECISIONS.md` with its time from `date`.
