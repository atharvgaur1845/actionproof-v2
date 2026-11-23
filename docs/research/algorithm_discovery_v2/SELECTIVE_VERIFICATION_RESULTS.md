# Selective verification results

**Testbed.** TB2 rich workflows, five seeds. These are development results registered in V2TB_PROTOCOL; the
data are `results/ad2/tabular/TB2_summary.json` and `TB2_stats.json`.
- **Op types:**
  - order: p 0.08, repair 0.9, visible timeouts with duplicate risk;
  - receive: p 0.15;
  - pay: p 0.25, repair 0.5, false alarms 5%;
  - post: never faulted.
- **Workflows:** three, trained jointly.
- **Settings:** single and multi fault × c {0.02, 0.05, 0.1} × L_f {0, 1}, with duplicate loss D = 0.5.

**Metrics.** Net utility is U − cost − failure loss − duplicate loss. Regret is J* − J, computed exactly.

## 1. All metrics (means over settings, workflows and seeds)

| Variant | Episodes | Regret | Never-faulted-op verify | Risky-op verify | Duplicate loss | Precision | Recall | Uninformative reads | Recovery given silent fault | Cost per completed workflow |
|---|---|---|---|---|---|---|---|---|---|---|
| **igc ×8** | 51,200 | **0.0043** | **0.001** | 0.880 | 0.012 | 0.164 | 0.889 | **0.040** | 0.601 | 0.103 |
| awrbn ×8 | 51,200 | 0.0049 | 0.012 | 0.881 | 0.012 | 0.162 | 0.889 | 0.051 | 0.600 | 0.104 |
| rloo ×8 | 51,200 | 0.0051 | 0.010 | 0.879 | 0.012 | 0.164 | 0.887 | 0.051 | 0.600 | 0.104 |
| nostd ×8 | 51,200 | 0.0052 | 0.021 | 0.879 | 0.012 | 0.164 | 0.887 | 0.054 | 0.599 | 0.104 |
| igc ×4 | 25,600 | 0.0053 | 0.002 | 0.865 | 0.012 | 0.169 | 0.871 | 0.039 | 0.591 | 0.100 |
| nostd ×4 | 25,600 | 0.0080 | 0.039 | 0.850 | 0.011 | 0.160 | 0.855 | 0.068 | 0.580 | 0.099 |
| cvc k4 | 92,003 | 0.0093 | 0.020 | 0.845 | 0.012 | 0.158 | 0.847 | 0.054 | 0.575 | 0.098 |
| cvc k1 | 27,794 | 0.0103 | 0.020 | 0.827 | 0.011 | 0.161 | 0.831 | 0.052 | 0.563 | 0.094 |
| awrb ×1 (group std) | 6,400 | 0.0105 | 0.016 | 0.940 | 0.013 | 0.165 | 0.959 | 0.062 | 0.644 | 0.120 |
| critic_q ×8 | 51,200 | 0.0105 | 0.039 | 0.883 | **0.018** | 0.160 | 0.892 | 0.058 | 0.602 | 0.103 |
| fork_bel k1 | 32,645 | 0.0142 | 0.022 | 0.817 | 0.012 | 0.159 | 0.822 | 0.099 | 0.555 | 0.096 |
| **igc ×1** | 6,400 | **0.0155** | **0.020** | 0.800 | 0.012 | 0.164 | 0.812 | **0.052** | 0.545 | 0.093 |
| fork_act k1 | 32,635 | 0.0167 | 0.023 | 0.798 | 0.011 | 0.163 | 0.806 | 0.095 | 0.540 | 0.095 |
| awrbn ×1 | 6,400 | 0.0249 | 0.210 | 0.796 | 0.012 | 0.152 | 0.804 | 0.163 | 0.540 | 0.101 |
| **nostd ×1** (Dr. GRPO) | 6,400 | **0.0296** | **0.197** | 0.754 | 0.011 | 0.152 | 0.770 | **0.163** | 0.520 | 0.097 |
| critic_q ×1 | 6,400 | 0.0309 | 0.188 | 0.777 | 0.018 | 0.151 | 0.785 | 0.160 | 0.530 | 0.094 |
| critic_v ×1 | 6,400 | 0.0441 | 0.261 | 0.705 | 0.014 | 0.147 | 0.721 | 0.208 | 0.481 | 0.093 |
| lagr ×1 | 6,400 | 0.0752 | 0.163 | 0.566 | 0.011 | 0.158 | 0.588 | 0.147 | 0.393 | 0.068 |
| batchstd ×1 | 6,400 | 0.1727 | 0.006 | 0.033 | 0.002 | 0.144 | 0.035 | 0.117 | 0.023 | 0.010 |
| fork_bel_std k4 | 111,267 | 0.1783 | 0.002 | 0.003 | 0.000 | 0.102 | 0.002 | 0.175 | 0.001 | 0.008 |
| **grpo ×1** | 6,400 | **0.1834** | 0.001 | **0.002** | 0.003 | 0.098 | **0.002** | 0.178 | **0.001** | 0.006 |
| gdpo ×1 | 6,400 | 0.2057 | 0.001 | 0.001 | 0.018 | 0.078 | 0.001 | 0.280 | 0.001 | 0.000 |

**How to read the low-cost rows.** A low "cost per completed workflow" at low recall (grpo, gdpo, batchstd) is not
efficiency; it is non-verification. Their net utility is the worst (regret 0.17–0.21).

**Precision** is about 0.16 for every verifying policy. That is close to the base fault rate of the risky ops: the
optimal policy verifies every risky op, so precision equals the prior. Precision is not a selectivity measure here.
The never-faulted-op rate and the uninformative-read fraction are.

## 2. Does the policy follow the value of information?

**1. The standard estimator does not verify at all.** GRPO verifies risky operations at 0.002 and recovers 0.1% of
silent faults; batch std, GDPO and the fork-group std do the same. This is the vote rule
(GRPO_FAILURE_DERIVATION §3): every rich-workflow op has p·q < 1/2.

**2. Correct-sign estimators verify risky ops, but at 1× budget they are not selective.** Dr. GRPO verifies the
never-faulted op 19.7% of the time, and 16% of its reads carry no information. That is the testbed analogue of
first-stage F3, where the LLM verified the journal entry as often as payments.

**3. Exact local credit for uninformative reads (igc) makes the policy selective.**

| Metric (1× budget) | nostd | igc |
|---|---|---|
| Never-faulted-op verify | 0.197 | **0.020** (10× less) |
| Uninformative reads | 0.163 | 0.052 |
| Regret | 0.0296 | **0.0155** |

Gates T1–T3 pass at 1× and at 4×:
- **1×:** better in 5/5 seeds; paired difference +0.0141, CI [+0.0117, +0.0167], d_z 1.45, Holm p < 1e-4;
- **4×:** 0.0053 vs 0.0080, 5/5 seeds, Holm p < 1e-4.

At 8× the advantage shrinks to 0.0043 vs 0.0051 (16%, below the 20% gate; Holm p 0.18–0.29). Every correct-sign
estimator eventually reaches the same selective optimum. igc's benefit is **sample efficiency of selectivity**, not
a different end point.

**4. Counterfactual forks do not pay for themselves.** At matched episodes, cvc, fork_bel and fork_act are all worse
than rloo/nostd ×8, by −0.004 to −0.012 regret in 5/5 seeds. Their selectivity matches igc's, because they share
its exact credit on uninformative reads (cvc) or estimate it (forks). Their informative-decision forks add little
the sequence signal does not already give.

**5. Duplicates.** All correct-sign estimators have duplicate loss 0.011–0.013, except the learned Q-critic
(0.018), which retries blindly after timeouts more often. No candidate regresses duplicate prevention (T3).

**6. Recovery** given a silent fault is about 0.60 at convergence for every correct-sign estimator, against
GRPO's 0.001. The recovery skill itself is fixed by the environment's repair rates. The estimators differ only in
whether they look.

## 2b. Scope correction (TB7, V-F9)

About one third of igc's regret gain in TB2 comes from gating re-verification after a repair. In the testbed that
decision is labelled uninformative only because the replay holds the rollout's hidden repair-success bit fixed. The
artefact-free variant gates only the first read after a write (igc_v1):

| Budget | igc_v1 regret | nostd regret | Seeds better | igc_v1 never-faulted-op verify | nostd never-faulted-op verify |
|---|---|---|---|---|---|
| ×1 | 0.0202 | 0.0296 | 5/5 | 0.020 | 0.197 |
| ×4 | 0.0062 | 0.0080 | 5/5 | 0.002 | 0.039 |

**The selectivity effect is entirely in the first-read gate.** The headline numbers below should be read with this
correction.

## 3. What this does and does not show

**Shown, in a controlled testbed with an exact optimum:**
- standard GRPO verifies nothing whenever p·q < 1/2;
- Dr. GRPO verifies but is unselective early;
- information-gated exact local credit gives selective verification about 4× more sample-efficiently (igc ×1 is
  close to nostd ×4);
- forked counterfactual credit is not worth its cost.

**Not shown:**
- **Any LLM result for igc.** Its LLM implementation is first-stage D3 (`nostd_ema_irc`, tasks 1216 and 1217). It
  had not run at the time of writing, because the main session's tasks occupy the lanes. It also uses the EMA
  scale, which collapsed in 2 of 4 D2 runs.
- **Novelty of igc as an algorithm.** It is per-decision credit with an exact local value. Its only non-standard
  ingredient is the support-based uninformativeness test (LITERATURE_AND_NOVELTY_AUDIT §6).
