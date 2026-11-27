# Vendored code and data

Source repository: ActionProof. Source commit for all files below: `d19c2997f8bc09502978d009f49443913207b784`
(branch `worktree-agent-ade9f3ffa1cf1a8ac`, from the stage 3/4 line). Each source file below was also compared
byte-for-byte with the main checkout (branch `submission-candidate`, `113d458411d84a0d8cf10d799c0cc40e24227a50`)
and was identical.

| Vendored file | Source | Change |
|---|---|---|
| `actionproof_min/env.py` | `actionproof/integrity/sim/env.py` | none (verbatim) |
| `actionproof_min/env_s3.py` | `actionproof/stage3/env.py` | import path only |
| `actionproof_min/truth.py` | `actionproof/integrity/sim/truth.py` | import path only |
| `actionproof_min/runtime.py` | `actionproof/integrity/sim/runtime.py` | import paths only |
| `actionproof_min/agents.py` | `actionproof/integrity/sim/agents.py` | import path only |
| `actionproof_min/lite_format.py` | `actionproof/rollout/lite_format.py` | none (verbatim) |
| `actionproof_min/prompts.py` | `actionproof/bench/prompts.py` (`TOOL_RESULT_FORMAT`), `actionproof/integrity/sim/prompts.py` (`PROC_NEUTRAL`) | two constants only |
| `actionproof_min/tasks.py` | `actionproof/bench/generator_proc.py` (`ProcTask`, `load_tasks`) | subset; also reads `.json` lists |
| `actionproof_min/gate.py` | `actionproof/recovery_gate/gate.py` (RG1, final sprint) | import paths only; written by `scripts/vendor_gate.py`, checked by `tests/recovery_gate/test_space_vendor.py` |
| `actionproof_min/gate_env.py` | `actionproof/recovery_gate/env.py` | instance-level wrapper for the proc2 env; freshness rule off (`SPACE_RULES`): this simulator's reads have no `consistent` flag |
| `actionproof_min/episode.py` | `scripts/p1_eval.py` (`start`, `tool_text`, `run`) and `actionproof/rollout/episode.py` (`system_prompt`) | rewritten as a single-episode generator; see deviations |

`scripts/p1_eval.py` is in the main checkout only (sha256 `5e230b42707853c2eda89e51802d0e3ea2aa5964570d36cb0001baa887efc3d8`).

## Parser

`actionproof.rollout.episode._agenttune()` tries `agenttune` (not installable on ZeroGPU) and falls back to
`actionproof/rollout/lite_format.py`. The Space vendors that fallback verbatim as `lite_format.py`, so it is the
same parser the repo uses wherever AgentTune is absent. Tests check that its output equals `_agenttune()`'s parser
on sample strings and that the fallback prompt text is identical. Where AgentTune is installed, the same test
compares against AgentTune's own parser. Parity on real model outputs is checked on the cluster by
`scripts/check_lite_parity.py`; that check was not re-run for this Space.

## Tasks

`data/tasks.json` holds 6 tasks copied unchanged from `data/bench/proc/p1_test_full.jsonl`
(sha256 `3ffd31614eb48001213bf364bfaaa839a65e1634d9e87055405a6c42edfae99c`). Selection rule: among tasks whose
`meta.faults` is empty and `never_commit` is null (so the only fault in a live run is the one the reality adds),
sort by `task_id` and take the first two of each workflow, in the order `order_prepay`, `receive_pay`, `pay_post`.
Result: `p1_test_full-0004`, `-0010` (order_prepay), `-0006`, `-0040` (receive_pay), `-0003`, `-0007` (pay_post).
Every workflow in the split includes `pay`. Selection used no model outputs.

## Realities

Built by replacing `meta.faults` (and clearing `never_commit`) with a single fault on the first `pay_supplier`
call: none; `phantom`; `partial` with `ratio` 0.5; `lost_ack`. The environment is `S3ProcEnv` (`make_env_s3`), as in
`p1_eval.run`.

## Recovery Gate in `episode.py` (final sprint)

`stream_episode(..., gate_rules=...)` puts the gate in front of the tools, yields a `gate_call` event for every call the gate makes itself, finalises an episode that ended without a report, and attaches `rec["gate"]`. With `gate_rules=None` (every pre-existing option) the loop is unchanged.
