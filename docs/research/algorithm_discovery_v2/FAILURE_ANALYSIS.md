# V2 failure analysis

This file records every failed hypothesis, failed run and negative result, with its diagnosis. Entries are not
removed. First-stage failures F1–F6 are in `docs/research/algorithm_discovery/FAILURE_ANALYSIS.md`.

## V-F1. H1c failed: resampling the hidden world per rollout does not remove GRPO's distortion (TB1)

| Field | Content |
|---|---|
| Registered | H1c: grpo under condition B (i.i.d. world per rollout) has a wrong-decision rate ≤ 0.15 |
| Actual | 0.315. grpo under condition A: 0.644; nostd: 0.057 |
| Diagnosis | The group std is still estimated from the same 8 samples. Their *composition* of hidden states varies, so 1/σ̂ still correlates with the realised worlds. A group in which no rollout hit a fault (probability (1 − p)^8, for example 0.43 at p = 0.1) has σ̂ ≈ c·sd(reads) and casts a full-size anti-read vote. The numerical demonstration, run after registration but before any TB1 output was read, predicted this: condition-B thresholds of 0.24–0.32 against VOI thresholds of 0.01–0.2. Pattern in TB1 (c = 0.05, s = 1): grpo_B verifies at 0.84 at p = 0.4 but at 0.01 at p = 0.2, where VOI = 0.15 |
| What it falsifies | The informal first-stage claim, and our own registration, that the *shared* world is the cause. The correct statement is the normaliser-independence condition (GRPO_FAILURE_DERIVATION §4). The shared world is the extreme case of a scale that correlates with the hidden state. Condition B weakens the correlation but does not remove it at G = 8 |
| What it does not falsify | The vote mechanism. Condition B halves the error, as the derivation predicts |

## V-F2. H1e failed narrowly: GDPO verifies in 11% of positive-cost cells, not below 10%

| Field | Content |
|---|---|
| Registered | gdpo never verifies (prob < 0.1) in ≥ 90% of cells with c ≥ 0.01 |
| Actual | 0.89. Its wrong-decision rate is 0.747, the worst of all 18 arms; its regret is 0.188 |
| Diagnosis | In high-p cells (p = 0.8, s = 1; verify prob 0.29 at c = 0.05), most groups are faulted and the per-component utility vote can outweigh the cost vote. That is consistent with the derivation's "unless almost every group is an outcome group" |
| Verdict | The hypothesis as registered fails by 0.01. The substantive prediction holds: GDPO is the most cost-biased estimator tested. Both statements are reported |

## V-F3. V2-E1 task 1300 failed without training (DV-2)

The cause is the bash builtin `GROUPS` being overwritten in `slurm/ad2_arm.sh`. E1 was then suspended by the
main session's GPU reprioritisation, so condition B was tested on CPU only (TB1, TB6). Because of V-F1, an LLM
condition-B run would now test a *weaker* prediction than the one registered: less collapse, not no collapse.

## V-F4. First-stage AWRB's development advantage is over-verification (AWRB audit)

awrb's +0.034 VSC over nostd on cw_dev comes with lower J_c in both seeds, 2.61 vs 1.74 reads per episode, and
0.91 verify-after-journal. The unbiased awrbn is below nostd on VSC and J_c in both seeds. TB1 reproduces the
mechanism: in the 18 "mirror" cells, where reading has negative VOI but p(1 + L_f) > c, awrb verifies in all of
them (1.0) and awrbn in none (0.0).

## V-F5. Every first-stage rollout used env and id seed 0 (audit finding, not a failed experiment)

Harmless for AWRB, because replay uses the same seed. Recorded because the code reads as if each rollout had its
own seed.

## V-F6. Counterfactual forks (fork_act, fork_bel, cvc) fail at matched budget (TB2, TB3)

| Field | Content |
|---|---|
| Hypothesis | Paired forked continuations at verification decisions give decision-level credit worth their extra rollouts |
| Actual | Gate T1 fails for every fork variant in 5/5 seeds against rloo/nostd ×8 at matched episodes (TB2: cvc k1 0.0103, fork_bel k1 0.0142, fork_act k1 0.0167, vs rloo ×8 0.0051). In TB3 every fork method has 2–3× the test regret of 4×-trained cheap estimators |
| Diagnosis | (i) The fork term matters only at *informative* verification decisions. There the sequence-level, correctly signed advantage already carries the right sign, and spending the episodes on more independent groups reduces variance as much. (ii) Selectivity comes from the *uninformative* decisions, which igc handles exactly at zero cost; cvc inherits igc's selectivity and adds fork cost. (iii) With K = 1–4 forks, the Q-difference at informative decisions is noisy (repair success 0.5 at pay) |
| Targeted diagnostic | At equal *updates* (TB2 ×1): cvc 0.0103 vs igc 0.0155, so forks help per update. At equal *episodes* (igc ×4, 25.6k, vs cvc k1, 27.8k): igc 0.0053 vs cvc 0.0103. The fork advantage per update is smaller than the gain from about 4× more groups |
| Falsifies | Forked counterfactual credit as a cost-effective method in this testbed. In the LLM, forks are partial continuations with shared prefixes, so the cost ratio could differ. That was not tested |

## V-F7. Per-context EMA scaling (ctxema) is not better than no scaling (TB2, TB3)

ctxema is sign-preserving, as derived, but worse than nostd at every budget in TB2 (e.g. ×4: 0.0093 vs 0.0080, 5/5
seeds negative) and at ×1 in TB3.

Diagnosis: with Adam, the global scale of the advantage is irrelevant. A per-context scale only *re-weights
contexts*, and here it over-weights low-variance contexts. The curvature argument of arXiv 2601.23135 does not carry
over to a setting where low reward variance means "verification only costs".

## V-F8. AD-D2 (first stage, LLM): the EMA-scaled no-std arms fail G1, and one seed of each collapses

nostd_ema s1 and awrbn_ema s0 end with 0 reads (VSC 0.707), the same as grpo, even though their scale does not
depend on the hidden world.

Diagnosis (not tested further): the EMA restores GRPO-sized steps, about 1/σ ≈ 6–8× nostd's. With a sign-correct but
noisy signal, larger steps can drive the read probability into the absorbing ρ = 0 state. No estimator recovers
from that state, because every group then has zero variance. This is a stability failure, not the vote mechanism.

It also qualifies first-stage D3 (nostd_ema_irc), which uses the same EMA scale.
