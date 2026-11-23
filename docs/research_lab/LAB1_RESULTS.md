# LAB1 results: counterfactual decision-value credit in the exact testbed (registered; development level)

**Run.** Slurm job 467649 (compute, 32 CPUs, node15), commit 0a1848f. The freeze check passed both on the cluster and
on the pulled-back files.

**Sources.** Every number is from `results/lab/lab1_summary.json`, `results/lab/grad_snr.json` and
`results/lab/lab1_refs.json`. Each analysis was run once, exactly as registered (`PREREGISTRATION_LAB1.md`).

**Evidence level.** This is a synthetic testbed with exact regret, registered before output and run on 5 seeds. It
is development evidence for an algorithm. It says nothing about LLMs.

## 1. Registered outcomes

| ID | Prediction | Result | Verdict |
|---|---|---|---|
| H1 | cdvo_adapt@0.25 vs SFB_np at E4 | 0.0170 vs nostd 0.0080 (−114%); better in **0/5** seeds; Holm p 1.0 | **not supported** (predicted) |
| H2 | cdvo_ia@0.25 vs SFB_all at E4 | 0.0097 vs igc_v1 0.0062 (−57%); better in **0/5** seeds; Holm p 1.0 | **not supported** (predicted) |
| H1, H2 at E1 | reported | cdvo_adapt 0.0595 vs ppo 0.0215; cdvo_ia 0.0380 vs igc_v1 0.0202; both 0/5 | not supported |
| H3 | CRN lowers fork-gradient variance on the risky verify logits at both θ | variance ratios: warm 1.12 (order), 0.98 (receive), 1.08 (pay); mid 0.94, 1.01, 0.98 | **not supported** (predicted supported; failure analysed in §3) |
| H4 | indep_world_std wrong rate ≥ 0.30 and indep_task_std ≤ 0.15 | **0.667** (44/66) and **0.061** (4/66); grpo 0.727, nostd 0.045 | **supported** |
| H5 / T4 | generalisation of a supported candidate | no candidate supported. Descriptively, both CDVO arms are worse than PPO on test_shift (0.0089 and 0.0065 vs 0.0028) and on test_long (0.0574 and 0.0474 vs 0.0243) | fail |

## 2. Main table (seed means over 12 settings × 3 workflows; exact regret)

| Arm | Budget | Regret | Never-faulted verify | Risky verify | Duplicate loss | Recovery after silent fault | Uninformative reads | Episodes | Updates |
|---|---|---|---|---|---|---|---|---|---|
| grpo | E1 | 0.1834 | 0.001 | 0.002 | 0.0030 | 0.001 | 0.178 | 6,400 | 200 |
| nostd (Dr. GRPO) | E1 / E4 | 0.0296 / 0.0080 | 0.197 / 0.039 | 0.754 / 0.850 | 0.0110 / 0.0113 | 0.520 / 0.580 | 0.163 / 0.068 | 6,400 / 25,600 | 200 / 800 |
| rloo | E1 / E4 | 0.0296 / 0.0080 | 0.194 / 0.042 | 0.754 / 0.850 | 0.0110 / 0.0113 | 0.519 / 0.580 | 0.162 / 0.069 | | |
| ppo | E1 / E4 | **0.0215** / 0.0103 | 0.103 / 0.021 | 0.785 / 0.839 | 0.0138 / 0.0139 | 0.539 / 0.573 | 0.125 / 0.071 | | 800 / 3,200 |
| ppo_norm | E1 | 0.0512 | 0.048 | 0.531 | 0.0118 | 0.365 | 0.109 | | 800 |
| critic_v | E1 / E4 | 0.0441 / 0.0103 | | | | | | | |
| critic_q | E1 / E4 | 0.0309 / 0.0131 | | | 0.0184 / 0.0182 | | | | |
| igc_v1 (privileged support) | E1 / E4 | 0.0202 / **0.0062** | 0.020 / 0.002 | 0.787 / 0.872 | 0.0122 / 0.0120 | 0.539 / 0.594 | 0.045 / 0.038 | | |
| awrb / awrbn | E1 | 0.0105 / 0.0249 | 0.016 / 0.210 | 0.940 / 0.796 | | | | | |
| fork_act (full branching) | E1 / E4 | 0.0942 / 0.0216 | 0.119 / 0.033 | 0.504 / 0.779 | | 0.340 / 0.527 | | 6,505 / 25,681 | 40 / 157 |
| fork_act_nocrn | E1 / E4 | 0.0990 / 0.0224 | | | | | | | 40 / 157 |
| cdvo_fixed@0.25 | E1 / E4 | 0.0601 / 0.0143 | 0.291 / 0.114 | | | | | | 99 / 395 |
| cdvo_adapt@0.25 | E1 / E4 | 0.0595 / 0.0170 | 0.281 / 0.179 | 0.654 / 0.815 | 0.0133 / 0.0105 | 0.447 / 0.555 | 0.244 / 0.134 | | 87 / 347 |
| cdvo_adapt@0.10 | E1 / E4 | 0.0462 / 0.0130 | | | | | | | 126 / 511 |
| cdvo_adapt_critic@0.25 | E1 / E4 | 0.1091 / 0.0851 | 0.077 / 0.016 | 0.301 / 0.369 | 0.0076 / 0.0050 | 0.211 / 0.259 | | | 89 / 363 |
| cdvo_ia@0.25 | E1 / E4 | 0.0380 / 0.0097 | 0.051 / 0.006 | 0.720 / 0.831 | 0.0123 / 0.0112 | 0.488 / 0.567 | 0.061 / 0.039 | | 105 / 426 |

**References** (`lab1_refs.json`):
- Bayes-optimal: regret 0 by definition, mean J* = 0.720.
- Scripted policy (verify every risky write, repair on a flag): regret 0.0064. Only igc_v1 at E4 (0.0062) is below it.

**Reproduction of v2.** The shared arms match v2 exactly at the same settings and seeds: nostd E1 0.0296 (TB2 ×1),
igc_v1 E1 0.0202 (TB7), awrb 0.0105, grpo 0.1834. The trainer is bit-identical (`test_reproduces_v2_trainer_exactly`).

## 3. Mechanistic analysis

**Exact gradient study (L2G, 300 batches per estimator and θ, rich multi c 0.05 L_f 1).**
- **GRPO has the wrong sign** on all three risky verify logits (order, receive, pay), at the warm start and at the
  undecided policy. The cosine with the exact gradient is −0.175 (warm) and −0.119 (mid). This is the vote rule in a
  multi-decision, multi-fault, timeout and false-alarm workflow, beyond Theorem 1's single decision.
- Every other estimator has cosine ≥ 0.989 and the right sign on every entry.
- **Where variance falls.** On the never-faulted operation's logit, SNR is 0.34 for nostd, 12.4 for fork_act and
  23.5 for igc_v1. On the risky logits, forks barely move SNR:

  | Logit | fork_act SNR | nostd SNR |
  |---|---|---|
  | order | 0.14 | 0.12 |
  | receive | 0.38 | 0.35 |
  | pay | 0.46 | 0.33 |

  Decision-value credit removes variance only where the decision is *uninformative*, and there the exact value −c is
  free (igc). Where the read matters, the dominant variance is Var_z(Δ): faulted and clean worlds have opposite-sign
  Δ. Paired forks inside one world cannot remove it (CDVO_THEORY Prop. C2). Forks cost 5.1× (full) or 1.9–2.3×
  (b = 0.25) the episodes per policy update (E4: 157, and 347–426, vs 800 updates). The gain per update is smaller
  than the loss of updates.
- **Why H3 failed.** CRN couples only the exogenous draws after the fork. The verify branch adds an observation, so
  the two branches consume the random stream differently and de-synchronise. The variance it could act on
  (E_z Var(Δ̂ | z)) is small compared with Var_z(Δ). Variance ratios of 0.94–1.12 are consistent with no effect. In
  training, CRN was better in 5/5 seeds at E1 (0.0942 vs 0.0990) and 3/5 at E4, both differences small and
  descriptive.

**Adaptive branching is not better than random branching.**
- E4: adapt 0.0170 vs fixed 0.0143, adapt better in 0/5 seeds. E1: 0.0595 vs 0.0601, 2/5.
- Fewer forks are monotonically better: b = 0.10 beats b = 0.25 in 4/5 (E1) and 5/5 (E4) seeds.
- Allocating by sign uncertainty sends forks to contexts with Δ ≈ 0. These are by construction the decisions whose
  correct credit matters least (gradient ∝ π(1 − π)Δ). Meanwhile the never-faulted operation's uninformative reads,
  where forks reduce variance most, are signed confidently after a few forks and stop being forked. The never-faulted
  verify rate of cdvo_adapt is the highest of all correct-sign arms at E4 (0.179). That is the exact opposite of the
  selective behaviour the method aimed for.

**A learned critic from stored fork means is harmful.** cdvo_adapt_critic has regret 0.085 at E4 and risky verify
0.369. Stale Q means (Prop. C3) keep the verify contrast at its early, low-recovery value. The policy locks into
under-verification. Its lower duplicate loss (0.0050) comes from doing less, not from being safer.

**Normalisation is the causal source (H4).** Two arms differ from grpo and nostd only in the scale:
- a scale computed from 16 **independent** rollouts in the same hidden world reproduces GRPO's failure (wrong 0.667
  vs grpo 0.727);
- the same construction in fresh worlds of the same task behaves like Dr. GRPO (0.061 vs 0.045).

The scale's *dependence on the hidden state*, not the reuse of the gradient's own samples, is what removes the cost.

**PPO's advantage normalisation.** ppo_norm (per-batch advantage normalisation, the usual PPO detail) is worse than
ppo in 5/5 seeds at E1 (0.0512 vs 0.0215). Its risky verify rate falls from 0.785 to 0.531. This is consistent with
the normaliser condition: a batch built from 4 hidden worlds. It is descriptive only.

**AWRB's reading.**
- In TB2 settings, awrb's high risky-op verification (0.940) is *beneficial* at E1: regret 0.0105, below nostd's
  0.0296. Here the optimum verifies every risky operation, so its over-verification bias points the right way.
- In the TB1 mirror cells (VOI < 0) it verifies in 18/18 cells against awrbn's 0/18 (v2 H1g), which is wasteful.
- So AWRB's bias helps exactly when verification is optimal anyway. It is not a decision-value estimator.

**igc: speed, not endpoint.** igc_v1 cuts never-faulted verification 10× at E1 (0.020 vs 0.197) and has the lowest
E4 regret. Its whole gradient advantage is on uninformative decisions, where its SNR is 23.5 vs 0.34. On the risky
logits its SNR equals nostd's. All correct-sign estimators share the same fixed point in this tabular class, and v2
TB2 ×8 converges (0.0043 vs 0.0052). igc therefore buys sample efficiency on wasted reads, not a different end point.

**Downstream recovery.** Recovery after a silent fault tracks the risky-op verify rate in every arm: 0.580 for nostd
E4 vs 0.555 for cdvo_adapt E4. Decision-value credit did not change what agents do *after* a read. The repair
decision's own credit is the sequence term in every arm.

## 4. Go/no-go for CDVO (author's Part IX)

| Item | Status |
|---|---|
| Defensible difference from the nearest prior art | **fails** (RELATED_WORK §4: TRPO vine + COMA advantage; CRR; RSPI/EPIG for allocation) |
| Estimator justified | passes (unbiased; CDVO_THEORY §2) |
| Implementation tests | pass (48 lab tests + 20 v2) |
| Testbed supports the hypothesis | **fails** (H1, H2: 0/5 seeds at both budgets) |
| Beats the strongest simple baseline | **fails** |
| Improves net utility | **fails** (regret higher) |
| Avoids unnecessary verification | **fails** for cdvo_adapt (0.179 vs 0.039); passes for cdvo_ia (0.006) |
| No duplicate or false-success regression | passes (duplicate loss within +0.002) |
| Survives seeds | consistently negative, 0/5 |
| Generalises | **fails** (T4) |

**Decision: CDVO is rejected. No GPU task is registered for it** (PREREGISTRATION_LAB1 §5).
