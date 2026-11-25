# External benchmarks

2026-11-25. Licences and access are taken from the registrations, which checked them at the source. Statements about benchmarks not run come from the design constraints below; licences for those were not re-checked in this session, which had no web access.

## 1. Constraints any external benchmark must meet

- **Context.** Tiny Aya Global serves an 8,192-token context. That covers every trained policy in this repository.
- **Tool interface.** The policies were trained on ProcureSim's JSON tool-call format (`[{"tool_name": ..., "parameters": ...}]`) and its system prompt. A benchmark with a different format tests format transfer as much as capability.
- **No contamination.** We never train on an external benchmark, so every external score is held-out by construction.

## 2. Run, registered

| Benchmark | Registration | Subset | Result | Status |
|---|---|---|---|---|
| IFEval (google/IFEval, Apache-2.0) | `PREREGISTRATION_R1.md` (frozen) | 541 prompts, prompt-level strict | Every GRPO arm within ±0.03 of its warm start. State reward 0.000, recovery-start −0.004 (H2 non-inferiority supported) | **valid**: general instruction following |
| BFCL v4 (bfcl-eval 2026.3.23, Apache-2.0) | R1 | 11 single-turn Python categories, prompting mode | Non-live AST: state reward −0.002, recovery-start +0.001. Irrelevance abstention is a base-model property (0.079) | **valid**: single-turn function calling, not stateful execution |
| τ-bench retail (sierra-research/tau-bench @ 59a200c, MIT) | `PREREGISTRATION_R1_TAU.md`, descriptive, deviations D-R1T-1, D-R1T-2 | test, 115 tasks × 3 trials, Aya Expanse 32B as simulated user, ChatReActAgent | Floor regime: base pass^1 0.058. SFT −0.020 [−0.038, −0.006]; GRPO −0.006 and −0.004, both intervals include 0 | **valid but uninformative**: the only stateful multi-turn benchmark run, and the model is at the floor. Most successes involve no write action |

**Answer to "does RL improve externally relevant capability?"** No improvement is shown. The registered external result is **retention**: ProcureSim GRPO does not measurably cost IFEval or BFCL ability. On τ-bench retail, the only stateful benchmark run, a 3B model is at the floor, so no capability claim can be made either way. General tool use (BFCL, IFEval) and verified execution (ProcureSim) are reported separately and never combined.

## 3. Not run, and why

| Benchmark | Blocker | What it would take |
|---|---|---|
| BFCL multi-turn | Needs BFCL's executable backends and multi-step harness (R1 §scope) | Integrate the harness; prompt-mode multi-turn with the ProcureSim format |
| ToolSandbox | Stateful, which makes it the most relevant. Needs its own tool-calling interface and user simulator, and was not integrated | An adapter from ToolSandbox's tool schema to our JSON format. Pilot base vs D on a stateful-dependency subset, with an overflow check like τ-bench's |
| AppWorld | (From the benchmark paper, recalled, not re-checked this session.) Agents write Python against app APIs, and long contexts are typical. A different action space from trained JSON tool calls | Not meaningful for these checkpoints without retraining |
| BrowserGym / WebArena | Browser observation and action spaces; observations far exceed 8k tokens | Out of scope for a 3B, 8k-context tool agent |
| SWE-bench Verified | Code repair, not tool-state verification | Irrelevant to the claims |

**Next reproducible step:**
1. Integrate ToolSandbox through an interface adapter.
2. Register a descriptive evaluation (base, D, Dn) with a context-overflow stop rule copied from `PREREGISTRATION_R1_TAU.md` §pilot.
3. Run the pilot of 20 tasks before the main run.
