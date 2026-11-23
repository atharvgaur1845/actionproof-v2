# Failure analysis

Every failed or partly failed expectation in the programme, recorded as it happened. Development data unless marked.

## F1. The toy model of AWRB was wrong (AD-T0, 2026-11-23 11:2x)

| Field | Content |
|---|---|
| Expected | AD-T0's "awrb" arm reads correctly in every setting. |
| Actual | The arm replaced a *read* rollout's return by its expectation over all worlds. That is a forked-continuation Q-value, not the implemented estimator. |
| Evidence | CPU dry run on the login node (`scripts/ad/cpu_dryrun.py`): a read that reveals an unrepaired fault scores R̄ = 0, while a no-read rollout in the same group scores 0.69. |
| Cause | A modelling error in the toy, not in the code. Conditioning on observations collapses the posterior after a read. |
| Diagnostic | AD-T0b models the estimator as implemented. It found the mirror bias: AWRB with group std reads even when reading is worthless. |
| Conclusion | The registered awrb arm (group std) is biased. Amendment 1, before any GPU run, added awrbn (no std) and a cost intervention to test the bias. |
| Falsifies? | It falsifies "AWRB + GRPO's normalisation fixes the verification decision". It does not falsify AWRB with a world-independent scale. |

## F2. GRPO collapses verification under a read cost (AD-D1; seed 0, seed 1 pending)

| Field | Content |
|---|---|
| Expected | Registered P1: grpo's verify-after-pay ends below the warm start's. |
| Actual | As expected. Training reads 0.59 → 0.03 per episode; verify-after-pay 0.21 → 0.01; zero-variance groups 19% → 86%. Greedy cw_dev equals the warm start: 0 reads, VSC 0.707, faulted-task VSC 0.254. |
| Interpretation | This is a failure of the standard method, which is what the theory predicts. It is not a failure of a candidate. |

## F3. Selective verification was not learned by any arm (AD-D1 P2)

| Field | Content |
|---|---|
| Expected | Registered P2: verify-after-post (zero value under cw1) falls in awrb and nostd. |
| Actual | It tracks verify-after-pay in every arm, in training and at evaluation. Seed 0, final training block: nostd post 0.49 vs pay 0.52; awrb post 0.76 vs pay 0.77. Evaluation: awrb post 0.91, pay 0.92. |
| Candidate causes | (a) **Credit smear.** The sequence-level advantage gives a valuable pay read and a worthless post read in the same trajectory the same credit. The only signal separating them is between trajectories, and it is worth c = 0.05. (b) **Representation.** The policy has learned "read the record after a write", not a site-conditional rule. (c) **Steps.** 200 steps may be too few. |
| Distinguishing diagnostic | Not yet run. Candidates: a per-read information-gated credit, which routes each read's cost to its own tokens when the replay posterior shows the read could not change the world posterior; or a 400-step run. |
| Caution | The main session's C5 result (2026-11-23) shows that per-turn credit normalised across turns gives reads *negative* credit by construction (their u = 0 for read turns), and C5 policies stopped reading altogether. Any per-read credit must avoid normalising read turns against write turns. |
| Falsifies? | It falsifies the claim that the corrected estimators produce *selective* verification within 200 steps. It does not falsify the VoI sign. |

## F4. Rao-Blackwellisation reduces within-group reward spread only modestly (AD-D1 seed 0)

| Field | Content |
|---|---|
| Expected | AD-T0b: an SNR gain when repair succeeds often. |
| Actual | Mean within-group std of the Rao-Blackwellised return R̄ − c·reads is 10–20% below that of the raw return (`analyze_d1.py` blocks). The mean within-group variance of R_V falls to 0.66–0.75 of raw over all 800 training groups of each seed-0 arm, so world luck is about 25–34% of within-group variance. Even so, awrbn learned to read *more slowly* than nostd in seed 0. |
| Cause (hypothesis) | Most within-group variance is action noise, not world noise. Once agents read, the world is revealed and R̄ = R. AD-T0b's caveat also applies: replacing no-read returns by the prior mean breaks the shared-world pairing. Repair after a revealed payment fault is low (awrbn 0.29), the regime where AD-T0b predicts no SNR gain. |
| Diagnostic | D2 (awrbn_ema vs nostd_ema at a matched scale). |

## F5. Every costed arm is below the free-read-trained baselines on cw_dev, even on J_c (context, not a gate)

| Field | Content |
|---|---|
| Observation | RSC s0: VSC 0.960, J_c 0.812. A12 R_V s0: VSC 0.913, J_c 0.769. Best costed arm at seed 0: J_c 0.740 (nostd). |
| Cause | Repair after a revealed payment fault: RSC 0.82, A12 R_V 0.61, nostd 0.53, awrb 0.44, awrbn 0.29. The free-read models were trained on fault-dense pools (and RSC on recovery starts), where repair is learned. Under cw1, 60% of training tasks are clean, so repairs are practised less. |
| Implication | Within the costed testbed, the estimator comparison is clean. Across training recipes it is not. **The costed arms do not beat the strongest existing ActionProof baseline.** Any claim must be scoped to "estimators under a costed, low-fault training distribution". |
