# Literature and novelty audit (v2)

Compiled 2026-11-23. This file extends `docs/research/algorithm_discovery/LITERATURE_AND_NOVELTY_AUDIT.md`
(entries L1–L12, not repeated) and the earlier `docs/research/NOVELTY_AUDIT.md` (84 entries).

**Verification labels:**
- **[abs]**: the arXiv abstract page was read this session;
- **[html]**: the full text was read through a summarising fetch, so equations are transcriptions;
- **[code]**: the implementation was read;
- **[std]**: a standard reference cited from knowledge and not re-read this session. Ids and venues are given as
  remembered. Do not cite one without checking it.

## 1. Primary sources on group normalisation

| # | Work | Read | Exact relevant content | Relation to our findings |
|---|---|---|---|---|
| N1 | Bereket, Leskovec. *Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes*. arXiv 2508.11800, 15 Aug 2025 | [abs]; first-stage [html] | Group-std normalisation makes GRPO overconfident for binary stochastic outcomes; removing it fixes this. PPO, RLOO and no-std GRPO are calibrated. Mechanism: policy-dependent 1/σ_x weights | **The mechanism is theirs.** Our Proposition 1 is their weighting with the hidden execution state in place of the stochastic outcome. New on our side: the consequence for an *information-gathering action* (the vote rule p·q > 1/2, cost-blind); the normaliser-independence condition; the counterexample; batch, fork and GDPO cases |
| N2 | Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO). arXiv 2503.20783 | first-stage [abs + html §3] | Removes the per-response length normaliser and the std normaliser ("question-level difficulty bias") | **No-std is their intervention:** a baseline, not a contribution. Our `nostd` arm removes the std only (loss aggregation unchanged) |
| N3 | Ge, Yin, Liang, Zhang. *Why GRPO Needs Normalization: A Local-Curvature Perspective on Adaptive Gradients*. arXiv 2601.23135 (v1 30 Jan 2026; v2 1 Oct 2026) | [abs] | Std normalisation acts as an adaptive gradient (local curvature); improved convergence over unnormalised REINFORCE under mild conditions; gains need heterogeneous per-prompt variances; IS-GRPO variant | **Complementary, and a constraint on any fix.** Removing the std gives up a real curvature benefit. Their analysis is per prompt and assumes the reward variance is a property of the prompt. Under a hidden state shared within the group, the within-group variance is a property of the *hidden state*, which the policy cannot condition on. Our condition (§4 of GRPO_FAILURE_DERIVATION) says where per-prompt normalisation stays valid: when the scale is independent of the current samples' hidden states. A per-observable-context EMA scale (`ctxema`) is the testbed's attempt to keep the curvature benefit without the bias |
| N4 | Bay, Yearick. *GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity*. arXiv 2607.00152 (30 Jun 2026) | [abs] | For binary rewards, the group std equals the update size: split groups teach most, unanimous groups nothing | Consistent with our "vote" reading. It does not discuss hidden state or costs |
| N5 | Liu et al. (NVIDIA). *GDPO: Group reward-Decoupled Normalization Policy Optimization*. arXiv 2601.05242 (8 Jan 2026) | [abs] + TRL 1.7.1 [code] (`multi_objective_aggregation="normalize_then_sum"`) | Normalise each reward separately within the group, sum, then batch-normalise | **We predict, and show numerically, that GDPO makes costly verification strictly worse.** It never verifies for c ≥ 0.05 (derivation §7; TB1 H1e). The cost component becomes an O(1) penalty in every group |
| N6 | Yang et al. *Your Group-Relative Advantage Is Biased*. arXiv 2601.08521 | first-stage [html] | Conditioning bias of the group-mean baseline on non-degenerate groups | Orthogonal |
| N7 | Gao, Shi, Hu, Harrison. *When Rewards Depend on the Group, GRPO Plays a Tournament*. arXiv 2611.02404 (2 Nov 2026) | [abs] | Group-dependent rewards: the group baseline changes the objective; Team-GRPO | Different mechanism (rewards depend on groupmates). The shared-hidden-state case is not covered |
| N8 | Sakib. *Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?* arXiv 2609.24144 (21 Sep 2026) | [abs] | Sharing a tool-fault noise schedule within a group (pairing) removes between-schedule reward-contrast variance, but need not reduce gradient variance. +5.1 pp under tool faults, 3 seeds; the registered learning-curve criterion was missed | **Closest empirical neighbour.** Paired = our condition A, unpaired = condition B. They study variance. We show that pairing + std creates a *bias* in the cost-benefit trade-off. Their abstract does not mention std or costs |
| N9 | TRL GRPO documentation (huggingface.co/docs/trl, main) and TRL 1.7.1 source | [html] + [code] | `scale_rewards` ∈ {group (default), batch, none}; "scaling by std(r) may cause a question-level difficulty bias … turning off std-based scaling also removes variance normalization"; `loss_type` default 'dapo'; Bessel std + 1e-4 (`nanstd`) | Exact denominator used in our reproduction (−134.3 vs the first stage's population-std −143.6) |

## 2. Critic-free and critic-based estimators

| # | Work | Read | Relation |
|---|---|---|---|
| E1 | RLOO: Kool, van Hoof, Welling, "Buy 4 REINFORCE samples, get a baseline for free!" (ICLR-W 2019); Ahmadian et al. *Back to Basics* arXiv 2402.14740 | [std] | Leave-one-out baseline, unbiased, no scaling. In the testbed as `rloo`. Sign-preserving by our condition |
| E2 | PPO (Schulman et al. arXiv 1707.06347); GAE (arXiv 1506.02438) | [std] | Learned value baseline. Testbed `critic_v` (tabular learned V over observable keys, Monte-Carlo advantage) |
| E3 | REINFORCE++ (Hu, arXiv 2501.03262): global batch normalisation | [std] | Testbed `batchstd`. Our condition predicts residual bias at small batch |
| E4 | DAPO (Yu et al. arXiv 2503.14476): dynamic sampling (drop zero-std groups), token-level loss | [std] | Dynamic sampling removes degenerate groups but keeps std. It does not fix the vote. TRL's 'dapo' loss is used by every LLM arm |
| E5 | Mean Actor-Critic (Allen et al. arXiv 1709.00503); Expected Policy Gradients (Ciosek and Whiteson arXiv 1706.05374) | [std] | All-action gradient Σ_a ∇π(a\|h)Q(h,a). Used by `critic_q`, `fork_*` and `cvc` |
| E6 | COMA counterfactual baseline (Foerster et al. arXiv 1705.08926) | [std] | Counterfactual over *other agents'* actions with a learned critic. Ours: counterfactual over the agent's own verification action, by simulation |
| E7 | Input-dependent baselines (Mao et al. arXiv 1807.02264) | first-stage [abs] | World-dependent baselines are unbiased; the justification for within-world centring |
| E8 | VinePPO (Kazemnejad et al. arXiv 2410.01679) | [std] | Monte-Carlo value estimates from intermediate states by re-sampling continuations: **the closest prior for `fork_act`**. They fork in the actual (fully observed) state |
| E9 | Tree-GRPO (Ji et al. arXiv 2509.21240, ICLR 2026); TreeRL (arXiv 2506.11902); ARPO | [abs] (Tree-GRPO); others [snip] | Branching agent rollouts with intra- and inter-tree group-relative advantages. **They normalise within shared-parent groups**, which by our condition reintroduces hidden-state dependence whenever branches share a hidden state. We could not verify from the abstract whether Tree-GRPO divides by a std |
| E10 | BiPACE (Wang et al. arXiv 2606.25556, Jun 2026) | [abs] | Action-counterfactual Q(s,a) − V(s) from peer steps clustered in the actor's hidden-state space; no extra rollouts. Closest *non-simulator* decision-level credit for LLM agents; not about hidden environment state |
| E11 | Asymmetric actor-critic (Pinto et al. arXiv 1710.06542) | [std] | Privileged critic, unprivileged actor: the class that AWRB, `igc` and `fork_bel` belong to |

## 3. POMDP, active sensing, value of information

| # | Work | Read | Relation |
|---|---|---|---|
| P1 | POMCP (Silver and Veness, NeurIPS 2010) | [std] | Root sampling: each simulation draws a state from the belief. **`fork_bel` is root sampling used to estimate a policy-gradient advantage at a decision point** |
| P2 | SAMPLR (Jiang et al. arXiv 2207.05219, NeurIPS 2022) | first-stage [abs] | Resample aleatoric parameters from the posterior given the trajectory |
| P3 | ACNO-MDP (Nam, Fleming, Brunskill, NeurIPS 2021); Act-Then-Measure (Krale et al. arXiv 2303.08271); AMRL-Q (Bellinger et al.) | first-stage [snip] | Costly observation in RL; "measure iff value > cost". AMRL-Q "converges to non-measuring policies": a different cause of a similar symptom |
| P4 | Calibrate-Then-Act (Ding, Tomlin, Durrett, arXiv 2602.16699, v3 May 2026) | [abs] + [html] (first 100k characters) | LLM agents trained with **GRPO** on correctness × cost discounts. "Its decision-making behavior collapses to a static policy" that always triggers unit tests. "RL without prior cannot internalize the structure." Their fix is to give the agent an inferred prior | **An independent observation consistent with our mechanism**: GRPO, costly information actions, collapse to a static rule. They attribute it to learnability, not normalisation. We have not verified their group construction or `scale_rewards`, so we do not claim it as an instance |
| P5 | Scores Are Not Decisions: CAM-DF (Feng et al. arXiv 2607.27083) | [abs] | Cost-aware stopping for tool acquisition, trained offline on the stop-vs-continue payoff gap (a decision-focused VOI label). Not RL |
| P6 | PSRL with LLMs (ICLR 2026, arXiv 2504.20997) | [snip] | Posterior sampling for exploration |

## 4. Constrained and risk-sensitive RL

| # | Work | Read | Relation |
|---|---|---|---|
| C1 | CPO (Achiam et al. arXiv 1705.10528); RCPO (Tessler et al. arXiv 1805.11074); Constrained GRPO and "Safety as a Budget" (earlier audit) | [std] / earlier audit | Lagrangian budgets. Under group std the multiplier cancels in cost-only groups (derivation §3d; testbed `lagr_grpo`) |
| C2 | CVaR policy gradient (Tamar et al. 2015); Risk-averse BAMDP (Rigter et al. arXiv 2102.05762) | first-stage | The first-stage `cvar` variant is not pursued |

## 5. Searches run this session (negative for an exact match)

- "GRPO standard deviation normalization bias shared environment state within group partial observability agent
  verification 2026" (extended). No paper on a hidden state shared within the group combined with std
  normalisation and verification cost.
- "LLM agent reinforcement learning learn when to verify tool call cost value of information selective
  verification 2026" (extended). This found P4 and P5 and SVRL (arXiv 2609.08025, a search-aware penalty); none
  analyses the normaliser.
- "posterior sampling hidden state counterfactual rollouts policy gradient LLM agent belief resampling fork
  advantage partially observable 2026" (extended). No belief-resampled fork advantage was found. Closest: BiPACE
  and PSRL-LLM.
- "tree search branching rollouts LLM agent RL intermediate state forks advantage" (standard). This found Tree-GRPO,
  TreeRL and ARPO.

Absence from four searches is weak evidence.

## 6. Novelty matrix for every candidate examined

| Candidate (testbed name) | Closest prior method | What is actually different | Privileged information | Novelty |
|---|---|---|---|---|
| No-std GRPO (`nostd`) | Dr. GRPO std removal (N2) | Nothing | none | **none (baseline)** |
| RLOO (`rloo`) | Kool et al.; Ahmadian et al. (E1) | Nothing | none | none (baseline) |
| Batch std (`batchstd`) | TRL "batch"; REINFORCE++ (E3) | Nothing. Our contribution is the prediction that it is biased for small batches | none | none |
| EMA scale (`ema`, first-stage D2) | PPO-style running reward normalisation | Nothing material | none | none |
| Per-context EMA scale (`ctxema`) | Running per-prompt statistics; N3's adaptive-gradient view | The scale is keyed by observable context and computed from *past* steps only, so it is independent of the current hidden states. A principled corollary of our normaliser condition, but mechanically a small change | none | **weak** |
| Lagrangian budget (`lagr`, `lagr_grpo`) | CPO/RCPO, Constrained GRPO (C1) | Nothing. Our contribution is the cancellation result under std | none | none |
| AWRB (`awrb`, `awrbn`) | Input-driven baselines (E7), SAMPLR (P2), CRN/paired rollouts (N8) | The return is integrated over observation-consistent hidden worlds by exact replay (first stage) | simulator + prior weights | moderate as an estimator. **Not superior** (AWRB audit) |
| Information-gated read credit (`igc`, first-stage D3) | Per-step credit methods; C5 (main session, failed) | An uninformative verification decision gets its exact all-action value −c, from replay of the *support* only, not normalised across turns | simulator + prior support | weak to moderate |
| Actual-world forks (`fork_act`) | VinePPO (E8), Tree-GRPO/TreeRL (E9) | Paired forced-action forks at verification decisions, with CRN, the all-action gradient, and **no** normaliser. VinePPO forks for value estimation in fully observed reasoning; here the observation that the actual world is a posterior draw is what makes it unbiased under partial observability | simulator snapshot | weak (an application of VinePPO plus MAC) |
| Belief forks (`fork_bel`) | POMCP root sampling (P1) + VinePPO (E8) + SAMPLR (P2) | Forks whose hidden world is drawn from the posterior at the decision, to estimate the belief-state Q-difference for a policy gradient. It resolves AWRB's trajectory-consistency limitation by re-generating continuations | snapshot + prior | weak to moderate: a combination of known pieces |
| Counterfactual verification credit (`cvc`) | igc + fork_bel | Exact local value where the read is provably uninformative (no forks); belief forks only at informative decisions | snapshot + prior | weak to moderate |
| Fork-group std (`fork_bel_std`) | Tree-GRPO-style per-node normalisation | A diagnostic, not a proposal | — | — |

**Honest summary.** No candidate is novel as a mechanism. The contributions that survive the novelty audit are
analytical:
- the vote rule;
- the normaliser-independence condition;
- the predictions for GDPO, batch std, constrained GRPO, condition B and tree/fork normalisation;
- the AWRB audit.

Any algorithmic claim must therefore rest on an *empirical* advantage of a combination (e.g. `cvc`) that is shown
under budget matching. Even then it should be described as a combination of known parts, not as a new algorithm.
