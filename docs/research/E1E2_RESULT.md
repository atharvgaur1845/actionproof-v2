# E1/E2 results (PREREGISTRATION_E1E2)

Computed 2026-11-23 09:54 IST by `scripts/e1e2_stats.py` (frozen; hash check passed), into `results/e1e2_summary.json`. Raw files:
`results/e1/` (21 models × 2 splits, greedy + 8 samples) and `results/e2/` (21 models × 480 contexts, greedy + 16 samples).
No model, run or task is missing.

## Registered outcomes

| Id | Result | Verdict |
|---|---|---|
| E1 P1: RSC − control (sampled VSC, 120 tasks) | +0.098 [0.081, 0.114], seed permutation p 0.052, Holm 0.103 | **Not supported** |
| E1 P2: RSC − ACK | +0.087 [0.075, 0.100], p 0.032, Holm 0.095 | **Not supported** |
| E1 P3: ACK − control | +0.010 [−0.003, 0.024], p 0.43, Holm 0.43 | **Not supported** |
| E1 T: threshold reading | Spearman ρ (greedy partial VSC on s3_conf_full vs E1 sampled VSC) = 0.92 over 20 runs. Highest sampled value of a greedy non-learner 0.161 ≥ lowest of a greedy learner 0.117 | **"Threshold"** by the registered rule |
| E2 H1: position (first action, S1V0) | Mean d = −0.013 [−0.028, 0.004]; negative in 13 of 20 runs | **Not supported** |
| E2 H2: reconciliation line, receive_pay | Mean +0.48 [0.43, 0.52] | **"Computation helps"** |
| E2 H3: after-refund re-pay (S2V0), descriptive | Correct re-pay rate by arm, order / pay-first / receive: RSC 0.40 / 0.12 / 0.29; control 0.35 / 0.13 / 0.31; R_V 0.25 / 0.09 / 0.23; ACK 0.25 / 0.07 / 0.18 | — |

**Sampled VSC per run** (warm start 0.032):
- RSC: 0.31, 0.28, 0.21, 0.31, 0.24;
- control: 0.30, 0.16, 0.07, 0.25, 0.08;
- ACK: 0.13, 0.20, 0.23, 0.07, 0.28;
- R_V: 0.12, 0.16, 0.12, 0.07, 0.03.

**Reading.**
- **Under sampling, the arms are continuous, not bimodal.** Sampled repair rates are low (0.03–0.31) and closely track greedy outcomes (ρ = 0.92).
- **Greedy decoding exaggerates the differences between runs.** That is the registered "threshold" reading. But the sampled ordering of runs is the same, so greedy outcomes are not noise.
- **RSC's advantage over control and ACK misses the Holm rule**, at five runs per arm.
- **A history-computable reconciliation line at the decision point more than triples correct refunds** in order-then-pay and receive-then-pay. It does not help in pay-first (next section).

## Exploratory (not registered; computed from the same raw files, `scratchpad e1e2_explore.txt`)

1. **Registered limitation.** E2 scores the first tool call of one turn, and a re-read counts as not correct. Re-reads are the most common first action in every cell, so E2 rates understate eventual repair. H1, as registered, does not measure the position effect.
2. **Position binding is in acting, not reading** (E1 full episodes, sampled).
   - After paying, agents read the payment record at similar rates in every workflow: 0.82–0.93.
   - Given that read, they refund much less often after a goods receipt:

     | Arm | Pay-first | Receive then pay |
     |---|---|---|
     | RSC | 0.69–0.80 | 0.32 |
     | Control | 0.64–0.66 | 0.26 |
     | ACK | 0.58–0.64 | 0.22 |
     | R_V | 0.33–0.52 | 0.08 |

   - In E2, posting immediately after the partial read is the most common decisive action after a receipt: 290 of 640 samples for RSC s0, against 107 and 167 pay-first.
3. **Re-paying with the original idempotency key** is a large part of the "dropped second step".
   - After the scripted refund in pay-first, RSC s0 re-paid with the original key in 288 of 640 samples and with a new key in 46.
   - In E1 episodes, refund followed by a same-key re-pay that is silently deduplicated (pay ABSENT) accounts for 3–16% of episodes, depending on arm and workflow.
4. **Paying the remainder.** With the reconciliation line, pay-first agents pay the outstanding remainder (354 of 640 for RSC s0) instead of refunding. ProcureSim's verifier marks two live payments as DUPLICATE, so this top-up scores as a failure. Whether a top-up should count as a valid repair is a benchmark-semantics question. It is recorded here, not changed.

These observations motivate the next registered experiments. They are not themselves claims.
