---
license: cc-by-nc-4.0
pretty_name: ActionProof traces
language:
- en
tags:
- agents
- tool-use
- evaluation
- reinforcement-learning
- procuresim
- actionproof
viewer: false
---

# ActionProof traces

The Hub's table viewer is switched off: the files mix several record schemas (episodes, summaries, splits, judge
cases), which it cannot combine. Browse them under *Files and versions*, or in the replay Space's episode explorer
(https://huggingface.co/spaces/atharvgaur18/actionproof).

Per-episode records, summaries, frozen task splits and judge-reliability cases behind the ActionProof results. Every
episode was run in **ProcureSim**, a simulated procurement back office (in-memory tables, ten tools, injected write
faults) and scored by a deterministic verifier. Nothing here comes from a real payment system or real suppliers.

- Code, pre-registrations, claims ledger: https://github.com/atharvgaur1845/actionproof-v2
- Models: `atharvgaur18/actionproof-agent-d` (recommended, merged), `-agent-d-adapters`, `-warmstart-dw`, `-agent-a`, `-agent-dn`.
- Recompute the published numbers from this dataset on a CPU: `bash scripts/verify_numbers.sh` in the code repository.

**This repository mirrors the code repository's paths**, so downloading it into a clone puts every file where the
analysis scripts expect it:

```bash
huggingface-cli download atharvgaur18/actionproof-traces --repo-type dataset --local-dir . --include "results/**/*.jsonl"
```

## Contents

| Path | What | How it was produced |
|---|---|---|
| `results/p1/test/<tag>__<split>.jsonl` | Episodes of every P1 arm on the frozen test: A, B, C, D, Dn (seeds 0-2, tags `A_s0` ...) and Dw (no seed). 5 splits x 5 episodes per task (1 greedy, 4 sampled) | `scripts/p1_eval.py --phase test`, one run per model and split (the script refuses to overwrite) |
| `results/p1/test/*.summary.json` | Aggregates of the file above (greedy and sampled, per workflow) | Same run, `summarise()` in `scripts/p1_eval.py` |
| `results/p1/dev/` | Development-split episodes and summaries (`p1_dev_full`, `p1_dev_rec`); used only to label the demonstration agent | `scripts/p1_eval.py --phase dev` |
| `results/p1/p1_summary.json` | Registered P1 analysis (B, C, D versus A; Holm; development winner) | `scripts/p1_stats.py` |
| `results/p1/auditkit/` | AuditKIT TaskCompletion judge scores on greedy seed-0 episodes (secondary, descriptive) | `scripts/p1_auditkit.py` |
| `results/p1/rollouts/` | Training rollouts of D seeds (gzipped JSONL) | GRPO rollout logs |
| `results/p2/{test,dev,ref,shift}/` | P2 arms (E, B2, Ag, Cg, Dg, Eg), references (W, class-matched control M, scripted oracle O) and shift-split episodes (Hindi, Hinglish, renamed API), with `provenance.json` (split hash, task ids) | `scripts/p2_eval.py`; `p2_summary.json` from `scripts/p2_stats.py` |
| `results/p3/p3_summary.json` | Registered P3 analysis (Dn versus D): a null result | `scripts/p3_stats.py` (inputs are the Dn and D files above) |
| `data/bench/proc/p1_*.jsonl`, `s4_*.jsonl` | The frozen task splits: `p1_train_{full,pp,rec}`, `p1_dev_{full,rec}`, `p1_test_{full,pp,rec,wf4,combo}`; shift splits `s4_*` used by P2 | `actionproof/p1/build.py`, `scripts/freeze_p1.py`; hashes in `results/p1_hashes.json` |
| `evaluation/judge_next/data/{cases,truth}.jsonl` | 497 judge-reliability cases (genuine and false completions, plus documented perturbations) and their ground truth | `evaluation/judge_next/build.py` from recorded P1 test episodes of D/A/B/C seed 0; labels follow from the verifier or the perturbation, never from a model |
| `results/p4/`, `results/p5/` | P4 (Dw_aug against Dw) and P5 (recovery SFT on four other Cohere bases) episodes and registered summaries | `scripts/p1_eval.py`; `scripts/p4_stats.py`, `scripts/p5_stats.py` |
| `results/b1/` | B1: prompted, untrained models on the P1 test (Aya Expanse 8B/32B; supplementary Qwen3-235B-A22B and gpt-oss-120b) | `scripts/b1_eval.py`, `scripts/b1_api_eval.py` |
| `results/txn/` | The locked TXN test (440 tasks): every scripted policy, prompted and trained agent; registered `stats__txn_test.json` | `scripts/txn_eval.py`, `scripts/txn_stats.py` |
| `data/rft/sel1_{sel,ctl}_rows.jsonl`, `data/rft/sel1_rows.stats.json` | The SEL1 training rows of both arms, as hashed before training (`results/sel1_hashes.json`, key `data`) | `actionproof/sel1/build.py` (task 1800) |
| `results/sel1/`, `data/bench/proc/sel1_*.jsonl` | SEL1 (selective escalation, registered): SEL and CTL seeds 0-2, their parents and scripted references on `sel1_test` and `txn_test`; the registered `sel1_summary.json`; the SEL1 splits (hashes in `results/sel1_hashes.json`) | `scripts/sel1_eval.py`, `scripts/sel1_policies_eval.py`, `scripts/sel1_stats.py` |

Splits (task counts): train full 240, pp 90, rec 300; dev full 90, rec 90; test full 120, pp 90 (unseen payment
fractions 0.4 and 0.6), rec 120, wf4 60 (an unseen four-step workflow), combo 60 (two faults) = 450 test tasks.

## Episode record schema

One JSON object per line in `results/p1/test/*.jsonl` and `results/p1/dev/*.jsonl` (written by `scripts/p1_eval.py`,
scored by `E1.S3E.episode_record_proc` and the verifier `actionproof/integrity/sim/truth.py`):

| Field | Type | Meaning |
|---|---|---|
| `task` | object | The task: `task_id`, `split`, `workflow` (e.g. `pay_post`), `prompt` (the ticket), `initial_state` (rows of `pos`, `payments`, `receipts`, `journal`, `supplier_orders`, `suppliers`, `escalations`), `meta` |
| `task.meta` | object | `fault_class` (e.g. `s3:partial:ack`), `faults` (list of `{k, kind, tool, ratio}` injected on the k-th call of a tool), `never_commit`, `po_id`, `requested` (steps asked for), `start_prefix` (scripted calls already executed for recovery starts), `start_step`, `start_type`, `env_seed`, `env_version`, `domain` |
| `trace` | list | Every tool call: `{idx, tool, args, obs, by}`; `by` is `agent` or the scripted prefix |
| `attempts` | list | Evaluator-only fault log per write: `{idx, tool, po_id, fault, committed, from_idempotency_store, obs_status}`; `committed` is what the database actually stored (`full`, `partial`, `none`, ...). The agent never sees it |
| `report` | object | The agent's `submit_report` (`status` completed / failed / escalated, `summary`), or null |
| `final_state` | object | Database rows after the episode |
| `score` | object | Verifier verdict: `steps` (per requested step: `VERIFIED_SUCCESS`, `PARTIAL`, `DUPLICATE`, ...), `collateral`, `collateral_rows`, `duplicate_irreversible`, `goal_met`, `vsc`, `false_success`, `r_v`, `r_a`, readback counters |
| `vsc` | int 0/1 | Verified-success completion: goal met in the database and no forbidden write. The primary metric |
| `sample` | int | -1 = greedy run; 0..3 = sampled run (T = 1, seed from task, sample and turn) |
| `task_id`, `rollout_id`, `task_prompt`, `fault_class`, `workflow`, `start_type` | | Convenience copies |
| `turns`, `policy_calls`, `n_model_calls`, `n_verify`, `invalid`, `ended_by` | | Episode length and ending (`report`, `no_call`, `context`, `max_turns`) |
| `conversation` | list | Full chat: system prompt (policy plus tool schemas), user ticket, assistant turns (`[{"tool_name": ..., "parameters": ...}]`), tool results (`Tool result for <name>: <json>`) |
| `violation`, `honest_fail` | bool | Duplicate or unrefunded write left behind; honest "failed"/"escalated" report on an unmet goal |
| `policy_reads`, `redundant_reads`, `refunds`, `repay_after_refund`, `repay_new_key`, `partial_pay_task` | | Behaviour metrics on the policy's own calls (after the scripted prefix) |
| `recon` | bool | True only for arm B (reconciliation line shown at inference) |
| `receipt_rollout_ids` | list | Internal ids tying the receipt ledger to the rollout |

`*.summary.json`: `{tag, split, phase, recon, model, adapter, samples, seconds, greedy, sampled, by_workflow_sampled}`;
`greedy` and `sampled` hold `n, vsc, false_success, duplicate_unrefunded, honest_fail, partial_pay_repaired,
new_key_share_on_repay, redundant_reads_per_episode, reads_per_episode, calls_per_success, context_ended`.
`model` and `adapter` are cluster-relative paths (`runs/...`) of the original runs. The P2 files add `provenance.json`
(`tag, split, phase, git_commit, split_sha256, task_ids`).

Task-split files (`data/bench/proc/p1_*.jsonl`): `{task_id, split, workflow, prompt, initial_state, meta}` (as `task`
above).

Judge cases: `cases.jsonl` = `{case_id, views}` with four views of one episode (`transcript`, `ledger`, `state`, `facts`);
`truth.jsonl` = `{case_id, category (A-Q), accept, completed, violation, abstain_ok_views, note, pair, source, sample, task_id}`.

## Licences

- **Episodes, summaries, judge cases:** CC-BY-NC-4.0. The agent turns in `conversation` and `trace` are generations of
  models derived from `CohereLabs/tiny-aya-global` (CC-BY-NC-4.0 plus the Cohere Labs acceptable-use policy), so the
  episode files inherit the non-commercial term. Prompted-model episodes added later (B1) come from the models named in
  their folder and carry those models' terms.
- **Task splits (`data/bench/proc/`):** generated by the repository's code (Apache-2.0, `LICENSE`) from synthetic data
  (invented suppliers, purchase orders, ids); distributed here under the same CC-BY-NC-4.0 as the rest of the dataset
  for simplicity. The code repository carries them under Apache-2.0.
- **Code:** Apache-2.0.
- No personal data. All names, ids and account numbers are synthetic.

## Limitations

- Simulator results, not production payments. A faulty acknowledgement here is an injected, labelled fault, not an
  observed real-world failure rate.
- The frozen test was run once per model; intervals in the analysis files are task-level (the trained models are held fixed).
- Generalisation fails on renamed tools and on the four-step workflow (`wf4`); see the model cards.
- P3 (no-std GRPO, Dn versus D) is a null result and is included for completeness.
- Arms C and D are not nested; the C-to-D gap does not isolate GRPO.

## Citation

```bibtex
@misc{actionproof2026traces,
  title        = {ActionProof traces: verified episodes of tool agents on ProcureSim},
  author       = {atharvgaur18},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/atharvgaur18/actionproof-traces}},
  note         = {Code: https://github.com/atharvgaur1845/actionproof-v2}
}
```
