# Counterexamples and scope conditions

Evidence: **[math]**, exact. Endpoints come from π₀ = 0.3, G = 8 (`results/theory/scan.json` → `counterexamples`,
`severity`) unless stated. "Endpoint 0 (regret r)" means the mean-field flow ends at never-verify with expected-utility
regret r.

**Classification**
- (1) invalidates the theorem;
- (2) requires a narrower statement;
- (3) changes a coefficient or finite-sample correction only;
- (4) irrelevant to the ActionProof implementation.

| # | Perturbation | Exact result | Class | Notes |
|---|---|---|---|---|
| 1 | **Non-binary stakes / severity** (L = 5, p = 0.3) | GRPO shared: endpoint 0, regret 1.75; iid 0.632, regret 0.645; Dr. GRPO 1 | (2) the vote holds (Thm 3.1); stakes are invisible by design | this *is* the result: severity cannot enter |
| 2 | Rare catastrophic fault (p = 0.1, L = 20) | GRPO 0 under shared **and** iid (regret 2.05); Dr. GRPO 1 | (2) | two fault classes: GRPO's update is unchanged when the rare state's stakes are multiplied by 1000 (−2.961 both); Dr. GRPO's goes −0.0147 → +57.26 (`severity.rescale_witness`) |
| 3 | Read cost above the stake (c = 1.2 > V) | every estimator ends at 0, correctly | (2) c enters through signs only | "irrespective of costs" needs c < stake |
| 4 | State-dependent cost (c = 0.01 clean, 0.4 faulted) | GRPO shared 0 (regret 0.173); iid 1 | (2) | the vote ignores which state is expensive |
| 5 | State-dependent cost reversed (0.4 clean, 0.01 faulted) | GRPO 0 (regret 0.017); **AWRB also 0** | (2) for AWRB's guarantee | AWRB's sign(ΔQ) property needs state-independent costs |
| 6 | Stochastic repair as rollout noise (p = 0.6, s = 0.6) | vote of signs = +0.2, yet GRPO shared ends at 0 (regret 0.31); iid 1 | **(2)**: the vote form of Cor. 3.2 does not apply; the threshold becomes 0.667, not 1/(2s) | the LLM is in this regime (repair is the policy's own later behaviour) |
| 7 | Stochastic repair as a state (split) | vote −0.28 = 2ps − 1; GRPO 0 | none: Cor. 3.2 exact | — |
| 8 | Imperfect detection (d = 0.7) as noise | GRPO shared 0 (regret 0.37) | (2) as #6 | — |
| 9 | False alarm + duplicate penalty (f = 0.2, D = 1) | GRPO shared 0 (regret 0.11); iid 0.502 | (2) | ΔQ ≤ net VOI if repair-on-alarm is not optimal |
| 10 | **Exogenous within-state noise** (q₀ = 0.3) | GRPO ends at **1**, correctly, under both laws | (2), important | noise not driven by the decision dominates σ̂, so standardisation stops cancelling the stake (Prop. 3.6). With enough unrelated variance GRPO becomes cost-sensitive again. The LLM sits between: 111/146 clean non-degenerate grpo s0 groups have constant R_V (cost-only variance) |
| 11 | Uninformative read (d = 0) | all estimators 0, correctly | none | cost-only state |
| 12 | K = 3 states (two faults, one stochastic) | GRPO shared 0 (regret 0.39); iid 0.695 | (2) | — |
| 13 | ε boundary: c = 0.001, 0.0005 | GRPO threshold 0.452, 0.412 (vs 0.5) | (3) | below 10⁻³ the ε floor decides; ActionProof c = 0.05 |
| 14 | Group size G | sign of the shared-law update G-independent; magnitude (G − 1)m_G | (3) under shared; (2) under partial sharing (non-monotone corner stability) | CORRELATION_EXTENSION §3.4 |
| 15 | **Independent states (condition B)** | GRPO still biased; interior π* (0.597 at p = 0.3) | (2): narrows "fixed by resampling states" to "reduced to Θ(1/G) regret" | Cor. 3.3 |
| 16 | GDPO-style per-component normalisation of (utility, −c·r) | never verifies under shared law, even at p = 0.6 (regret 0.55) and p = 0.95 | (2) extension: a separate cost component is a full anti-read vote | Prop. 4.3 |
| 17 | PPO clipping (multiple epochs) | first epoch on-policy (ratio 1): sign unchanged; later epochs clip magnitudes | (3) | TH1 `ppo` arm, descriptive. **ActionProof:** `num_iterations=1`, so (4) |
| 18 | KL penalty β to the warm start | adds a restoring term; endpoints move inside (0, 1) | (3) | TH1 `grpo_kl`; ActionProof β = 0.02 |
| 19 | Token-length normalisation (`loss_type="grpo"`, 1/\|τ_i\|) | positive per-rollout reweighting; signs preserved under the shared law | (3); (4) for ActionProof, which uses `dapo` | — |
| 20 | Several reward components summed before normalisation | Thm 3.1 applies to the sum | none | — |
| 21 | Correlated policy samples (shared prefix containing the decision) | Dr. GRPO coefficient × (1 − κ); κ = 1 removes the signal | (3) | — |
| 22 | Decision observed by the policy (not aliased) | results hold per observable history | (3) | A1 |
| 23 | Observation-dependent action availability | apply per history where verify is available | (3) | — |
| 24 | Off-policy / hinted data | identities need on-policy samples; hinted-episode importance ESS 1.01 / 322 (`results/quality/g_mismatch_s0_ck100.json`) | (4) for C1; blocks off-policy correction designs | — |
| 25 | Multiple decisions per trajectory | Thm 3.1 still exact; the vote is an approximation | (2) | LLM regime; LAB1 workflow testbed: GRPO's mean gradient is wrong-signed on every risky verify logit (cos −0.175, `results/lab/grad_snr.json`) |

**Nothing in this list invalidates the mathematics (no class (1)).** Items 1–2, 4, 12 and 15–16 are the substantive
predictions. Items 6, 8 and 10 are where the plain vote rule must not be quoted.
