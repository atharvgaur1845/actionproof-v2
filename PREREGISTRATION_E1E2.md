# Registration E1/E2: sampled outcomes and decision-point probes for partial-payment recovery

Registered before any E1/E2 output exists. No model is trained. Every model is an existing frozen adapter. Branch
`research/e1e2`. Motivation: `research_mandate/REPORT.md` §3 and §7 (descriptive trace diagnosis, not registered).

## 1. Questions

- **E1.** Under sampled decoding (T = 1), do partial-payment repair rates separate the arms by training run? Is the
  run-level bimodality seen under greedy decoding a threshold effect of greedy decoding?
- **E2.** At a fixed decision point, is the after-receipt failure:
  - a binding to workflow position (H1)?
  - a failure to compute the outstanding amount (H2)?
  - a dropped second step of the repair (H3)?

## 2. Data (new, hashed below; ids and PO ids disjoint from every existing split)

`scripts/e1e2_make_splits.py`, the same generator and fault dict as the addendum 2 targeted splits; partial ratios 1/4,
1/2 and 3/4 in turn.

| Split | Tasks | Content |
|---|---|---|
| `e1_pp_full` | 60 | Full episodes, 20 per workflow |
| `e1_pp_rec` | 60 | Recovery starts right after the faulted payment, ack and read alternating |
| `e2_pp` | 120 | Read starts, 40 per workflow |

These splits were never used for training, development or tuning. Each model is evaluated on them exactly once.

## 3. Procedures

- **E1** (`scripts/e1_eval.py`). The frozen Stage 3 episode loop: same prompt, parser, 16 turns, 256 tokens.
  - Each task is run once greedy and 8 times at T = 1.
  - Sampling uses the per-request seed crc32(task, sample, turn).
  - Endpoint: VSC.
- **E2** (`scripts/e2_probe.py`). One assistant turn per context: greedy once, plus 16 samples at T = 1 (seed crc32(task, condition)). Contexts:
  - **S1V0:** the read start as is.
  - **S1V1:** one reconciliation line appended to the read result, computed only from values visible in the conversation.
  - **S2V0:** S1V0 plus the scripted correct refund and its acknowledgement.
  - **S2V1:** S2V0 with the reconciliation line after the refund.
- **Correct first decisive action:**
  - S1: a refund of the partial payment's id, or an escalation. This is A15's strict rule.
  - S2: a full-amount `pay_supplier` with a new idempotency key, or an escalation. This is the scripted oracle's step.
  - Re-using the original key, paying only the remainder, posting, and reading are recorded but not counted as correct.

## 4. Models (21)

- `warm_start`;
- `a12_rv_s0-4`;
- `s3_rv_mix_s0-4` (RSC);
- `s3_matched_s0-4` (control);
- `s3_ackonly_s0-4` (ACK).

All adapters are on `runs/rft_proc/merged`, as in every S3 evaluation.

## 5. Hypotheses and rules (`scripts/e1e2_stats.py`)

**E1** (per-run sampled VSC over the 120 tasks):

| Id | Contrast | Rule |
|---|---|---|
| P1 | RSC − CONTROL | One-sided unpaired seed permutation (252); Holm over P1–P3; supported if Holm p < 0.05 **and** the seed-averaged 95% task-bootstrap interval excludes 0 |
| P2 | RSC − ACK | same |
| P3 | ACK − CONTROL | same |
| T | Threshold reading | Among the 20 trained runs, split by greedy partial-class VSC on `s3_conf_full` (≥ 0.40 = greedy learner): "threshold" if the highest sampled VSC of a greedy non-learner ≥ the lowest of a greedy learner, else "separated". Spearman ρ reported |

**E2** (per-run sampled correct rate per condition and workflow; 20 trained runs):

| Id | Statistic | Rule |
|---|---|---|
| H1 position | d = rate(S1V0, receive_pay) − mean of the two pay-first workflows | Supported if d < 0 in ≥ 16 of 20 runs (one-sided sign test p = 0.006) **and** the 95% bootstrap interval of mean d over runs is below 0 |
| H2 computation | g = rate(S1V1, receive_pay) − rate(S1V0, receive_pay) | "Computation helps" if mean g ≥ 0.10 with an interval above 0; else "not computation" |
| H3 chain | rate(S2V0) by arm and workflow; the H1 statistic on S2V0 | Descriptive |

**Inference limits, stated in advance:**
- Five runs per arm give a smallest one-sided permutation p of 1/252.
- Task intervals measure evaluation precision within runs, not uncertainty over training runs.
- E2 contexts are scripted. They test the policy at a decision point, not end-to-end behaviour.

## 6. Missing runs and deviations

- A model that fails to evaluate is re-run once. If it fails again, it is reported missing, and no contrast that needs it is computed.
- Any deviation is logged in `docs/stage2/DECISIONS.md` with its time taken from `date`.

## 7. Compute

- Four queue tasks (`slurm/tasks/1001`–`1004`) on the H200 lease or the A100. Each runs about 5–6 models.
- Estimated 1–2 GPU-hours per task.
