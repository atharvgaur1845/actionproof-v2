# Dataset audit and expansion

2026-11-25. Counts are computed from `data/bench/proc/*.jsonl` (generators: `actionproof/bench/generator_proc.py`, `actionproof/stage3/tasks.py`, `scripts/e1e2_make_splits.py`, `scripts/p1_*`, `scripts/p2_make_tasks.py`). Every registered split is hashed in its registration's `results/*_hashes.json`.

## 1. What exists (ProcureSim, procurement domain)

- **54 split files.** The registered evaluation splits are below.
- **Workflows:** three are trained (order → prepay, receive → pay, pay → post). One four-step workflow (order → receive → pay → post) is held out (`s4_wf4`, `p1_test_wf4`).
- **Never-commit tasks:** unreachable by design. The registered correct behaviour is an honest failed or escalated report.

| Family | Splits (tasks) | Faults present | Start types |
|---|---|---|---|
| A12 | proc_train 400, proc_train_b 400, proc_dev 200, proc_val 100, proc_conf 300, proc_stress 100, proc_combo 100, proc_heldout_service 100 | phantom, partial, lost_ack, precommit_timeout, never-commit; **replica_lag** and dup_delivery in stress | full |
| Stage 3 | s3_rec_train 400, s3_rec_dev 150, s3_conf_full 300, s3_conf_rec 300, s3_conf_stress 100, pp sets 150 + 150 | adds **dup_delivery** | full; recovery starts (ack, read, done) |
| Stage 4 shifts | s4_lang_en/hi/hinglish/para 150 each, s4_api 150 (renamed tools), s4_wf4 100, s4_combo 100 | all | full |
| E1/E2 | e1_pp_full 60, e1_pp_rec 60, e2_pp 120 | partial | full, ack, read |
| AD | cw_train 400, cw_dev 150, cw_conf 300 | all six | full |
| P1/P2 | p1_train_* 630, p1_dev_* 180, p1_test_* 450 (full 120, pp 90 with unseen fractions, rec 120, wf4 60, combo 60) | all; two-fault combos | full, ack, read, done |
| OpsSim (A17) | care_admin, IT operations | domain transfer | full |

## 2. Leakage audit

Prompt text was compared across every pair of splits.

**Training pool against evaluation split: exactly one overlap.**
- `cw_train` (the AD training pool) shares **75 prompts** with `c1_pp`, all full-episode tasks: same purchase order and text, different task ids.
- `c1_pp` is the Stage 5 confirmatory split. It was built and reserved, but **never used**, because no Stage 5 candidate survived development. It is also untracked in git.
- **Consequence:** `c1_pp` can no longer serve as held-out data for any AD-trained model. No reported result used it.
- **Correction:** the write-up's split list should drop `c1_pp` or state this.

**Every other overlap is a derived split by design:**
- half pools and their `_rb` / `_ackonly` variants are subsets of `s3_rec_train` or `proc_train_b`;
- the four `s4_lang_*` splits are the same tasks in four languages.

**Fault schedule exposure.** Prompts never mention faults. The fault schedule lives in `meta.faults`, which only the environment and verifier read. That is checked by the rollout code, which passes only `prompt`.

## 3. The brief's required splits, mapped to what exists

| Required | Exists as | Gap |
|---|---|---|
| train | proc_train_b, s3_trainb_half + s3_rec_train_half, p1_train_* | — |
| development | proc_dev, s3_rec_dev, cw_dev, p1_dev_* | — |
| validation | proc_val | used little |
| test_iid | proc_conf, s3_conf_full, p1_test_full | — |
| test_fault_ood | proc_stress / s3_conf_stress (dup_delivery, replica_lag) | Faults appear at training time in S3. No fully unseen fault class held out |
| test_tool_ood | s4_api (renamed tools) | — |
| test_domain_ood | OpsSim care_admin, IT ops (A17) | small |
| test_composition_ood | s4_wf4, p1_test_wf4 (unseen workflow), s4_combo, p1_test_combo (two faults) | — |
| test_adversarial | none in ProcureSim | **gap**. The judge study's synthetic categories (persuasive or forged reports) are judge-side, not agent-side |
| recovery_start | s3_conf_rec, p1_test_rec | — |

## 4. Faults in the brief: present or missing

| Fault | Status |
|---|---|
| granular partial commits | **present**: P1 unseen fractions |
| duplicate delivery | present |
| delayed acknowledgement | present as lost_ack (timeout reply, write committed) |
| stale reads | present as replica_lag (stress splits only) |
| already-completed states | present (`done` recovery starts) |
| unreachable tasks | present (never-commit) |
| idempotency-key conflicts | partial: a key reuse is deduplicated by design and the agent sees the original payment. No conflicting-payload case |
| out-of-order acknowledgements | missing |
| conflicting replicas | missing (replica_lag is one stale replica, not a conflict) |
| concurrent updates | missing |
| cross-service partial completion | missing |
| unauthorised operations, wrong target entities | the verifier detects them (collateral rows), but no task induces them |
| ambiguous instructions | missing |

## 5. Expansion: decision for this sprint

**No new ProcureSim tasks are generated before Nov 28.** Each missing fault needs a change to the fault chokepoint (`actionproof/integrity/sim/`). A changed verifier needs its own golden traces and reference-verifier extension, and a new registered evaluation. The brief forbids silently changing the verifier, and there is no GPU budget left to train against new faults.

**Reproducible next step**, for after the deadline. In order:
1. Extend `VERIFIER_SPEC.md` (proc3) with out-of-order acknowledgements, conflicting replicas and wrong-target writes.
2. Implement them in the fault chokepoint behind a version flag (`env_version`).
3. Add hand-built golden cases and extend `scripts/reference_verifier_proc.py` from the spec alone.
4. Generate `proc3_{train,dev,test_fault_ood,test_adversarial}` with a new seed, disjoint ids and disjoint PO ids. Check disjointness of prompts, not just ids: the `c1_pp` / `cw_train` overlap shows that ids are not enough.
5. Register an evaluation of the frozen P1/P3 models before running it.

**One cheap fix that applies now:** add a prompt-level disjointness check to every split builder. The overlap above passed the existing id-level check.
