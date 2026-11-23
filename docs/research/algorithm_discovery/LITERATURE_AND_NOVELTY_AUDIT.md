# Literature and novelty audit (Phase II)

Compiled 2026-11-23. This audit extends two earlier ones:
- `docs/research/NOVELTY_AUDIT.md` (84 entries verified 2026-11-21: recovery training, forked credit, curricula,
  constrained GRPO, stateful-tool RL);
- `docs/research/NOVELTY_MATRIX.md`.

It also draws on the mandate report's literature notes (`../research_mandate/lit/`). Those entries are referenced,
not repeated.

**Verification labels for entries new in this file:**
- **[abs]**: the arXiv abstract page was read;
- **[html]**: the method section of the HTML full text was read through a summarising fetch tool, so equations are
  transcriptions;
- **[snip]**: search snippets only. Not usable as a basis for a claim until read.

## 1. New entries most relevant to the candidates

| # | Work | Venue / id | Read | Exact relevant idea | Code | What our candidate adds |
|---|---|---|---|---|---|---|
| L1 | Bereket, Leskovec. *Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes* | arXiv 2508.11800 (Aug 2025), no venue shown | [html] | Binary stochastic outcome per prompt, shared by the group's scoring. With std normalisation, E[Â] ≈ Σ_x (1/σ_x) p_x (r − μ_x), with policy-dependent weights 1/σ_x, so GRPO becomes overconfident. Without std, E[Â] = ((G−1)/G) A. No-std GRPO, RLOO and PPO are calibrated. | not checked | **The mechanism is theirs.** Our part: (i) its consequence for a sequential agent's *information-acquisition* action: verification becomes cost-blind, with threshold 2p − 1 at perfect repair, and the mirror bias for Rao-Blackwellised returns; (ii) the tool-agent setting with hidden execution worlds; (iii) the AWRB estimator. |
| L2 | Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO) | arXiv 2503.20783 | [abs] + §3.1–3.2 [html] | Removes the 1/\|o\| length normaliser and the std normalisation ("question-level difficulty bias") | yes (oat) | Our `nostd` arm is the std removal only. Their motivation is difficulty weighting, not hidden worlds or costs. |
| L3 | Yang et al. *Your Group-Relative Advantage Is Biased* | arXiv 2601.08521 | [html], first 100k characters | The group-mean baseline conditioned on non-degenerate groups is biased by prompt difficulty; HA-DW reweighting. Groups are independent samples; no shared environment randomness. | not checked | Orthogonal: a different bias source. |
| L4 | Sakib. *Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?* | arXiv 2609.24144 (Sep 2026) | [abs] | Shares a fault schedule within a group (pairing) to remove between-schedule reward variance. Shows pairing "need not reduce gradient variance" and gives a counterexample. Tool-fault agent, 3 seeds. | linked | **Closest to AWRB's goal (remove world luck).** Opposite device: they share noise across a group; AWRB *integrates the hidden world out of each trajectory's return* by replay, with no extra rollouts. Their counterexample and ours (rb/none at s = 0) are the same family of caveat. The abstract does not mention std normalisation or verification costs. |
| L5 | Mao, Venkatakrishnan, Schwarzkopf, Alizadeh. *Variance Reduction for RL in Input-Driven Environments* | arXiv 1807.02264 (ICLR 2019 per the mandate's notes; the arXiv page shows no venue) | [abs] | Input-dependent baselines: condition the *baseline* on the exogenous input sequence; unbiased. | yes | AWRB conditions the *return* on the observation-equivalence class of the exogenous world (a Rao-Blackwellisation), not the baseline. It needs the prior only, and is exact through deterministic replay. |
| L6 | Jiang et al. *Grounding Aleatoric Uncertainty for Unsupervised Environment Design* (SAMPLR) | NeurIPS 2022, arXiv 2207.05219 | [abs] | Under curriculum-induced covariate shift, resample aleatoric parameters from the true posterior given the trajectory ("fictitious transitions") to keep optimality under the ground-truth distribution. | yes | Closest in spirit: a posterior over hidden parameters given the trajectory. SAMPLR samples one posterior world to fix curriculum bias in a value-based learner. AWRB takes the exact posterior expectation of the return for a group-relative policy gradient, to fix world luck and its interaction with group standardisation. |
| L7 | Mesnard et al. *Counterfactual Credit Assignment in Model-Free RL* | arXiv 2011.09464 (2020) | [abs, via search] | Future-conditional (hindsight) baselines that separate skill from luck; low variance. | — | Learned hindsight statistics vs exact simulator replay over a known exogenous prior. |
| L8 | Rigter, Lacerda, Hawes. *Risk-Averse Bayes-Adaptive RL* | NeurIPS 2021, arXiv 2102.05762 | [abs] | CVaR of return under epistemic (posterior over MDPs) and aleatoric uncertainty, via MCTS + Bayesian optimisation | — | Candidate C (posterior CVaR returns) is a per-trajectory, replay-based instance inside GRPO. Weak novelty. |
| L9 | Nam, Fleming, Brunskill. *RL with State Observation Costs in Action-Contingent Noiselessly Observable MDPs* | NeurIPS 2021 | [snip]; the PDF could not be parsed | ACNO-MDP formalism: pay to observe the state or act blind | — | Formal setting for costly verification. Their methods are planning and value-based, not group-relative LLM PG. |
| L10 | Krale, Simão, Jansen. *Act-Then-Measure* | arXiv 2303.08271; ICAPS 2023 | [snip] | QMDP-like heuristic: measure iff the measuring value exceeds the cost | — | The VoI rule we use as the optimum. Not an LLM method. |
| L11 | Bellinger et al. AMRL-Q; *Dynamic Observation Policies in Observation Cost-Sensitive RL* (survey) | arXiv 2307.02620 | [snip] | AMRL-Q "always converges to non-measuring policies" in stochastic settings | — | A different cause of the same symptom (a most-likely-state approximation), in tabular Q-learning. Worth citing as a prior observation of measuring collapse. |
| L12 | GiGPO; HGPO; Turn-PPO | NeurIPS 2025; arXiv 2512.17008 | in the earlier audit / [snip] | Step-level groups by repeated states; history-inconsistency bias; turn-level PPO | yes | Observation-keyed grouping merges aliased worlds (mandate §6.5). Not a fix for group-shared hidden worlds. |

Searches run this session, all with negative results for an exact match:
- "GRPO std normalization bias hidden state partial observability group shares environment world tool agent
  verification cost";
- "Rao-Blackwellized policy gradient exogenous noise conditional expectation return replay counterfactual worlds
  observation-equivalent";
- "LLM agent reinforcement learning learn when to verify tool calls verification cost budget GRPO 2026";
- "GRPO multi-turn agent rollouts group share same environment seed randomness bias";
- "hindsight relabeling hidden environment parameters policy gradient … LLM agent".

None returned a paper that analyses verification decisions under group-std GRPO, or that integrates the hidden world
out of an LLM agent's return by replay. Absence from five searches is weak evidence. A reviewer could know a paper I
did not find.

## 2. Candidate novelty table

| Field | A: AWRB (aliased-world Rao-Blackwellised returns) | B: no-std GRPO for costly verification | C: posterior-CVaR returns | D: certified invariance KL (mandate D) | E: C5 / G (existing, main session) |
|---|---|---|---|---|---|
| Closest prior work | L4, L5, L6, L7; PEGASUS/CRN; mandate candidate A | L1, L2 | L8; CVaR-PG (Tamar et al. 2015) | DrAC, StateAlign, ToolRL-DR | MT-GRPO, GiGPO, RewardFlow / PSP, Guide-GRPO |
| What prior methods do | Share noise across a group (L4); condition baselines on exogenous inputs (L5); resample posterior worlds for curricula (L6); learn hindsight baselines (L7) | Drop the std for difficulty bias (L2) or calibration (L1) | CVaR over the posterior over MDPs, by planning | KL invariance under augmentation | Turn credit; sampler-only hints |
| Our exact difference | R̄(τ) = Σ_w π₀(w) 1[o(τ,w) = o(τ)] R(τ,w) / Σ_w π₀(w) 1[…], by deterministic replay of τ's own calls in every prior world, with zero extra generation; used as the GRPO return | Applies the known fix to a new failure (cost-blind verification) and tests it | Same replay with CVaR_α in place of the expectation | Simulator-certified pairs | none claimed (C5G_DEV.md) |
| Novelty | **Moderate** as an estimator; the domain-specific consequence analysis is **moderate**. The general Rao-Blackwell idea is classic. | **Not novel** as a method. The analysis of its consequence is shared with A. | Weak | Weak to moderate | Not novel |
| Overlap risk | A paper that integrates exogenous noise out of returns by simulator replay in LLM RL; SAMPLR-style return grounding | — | Rigter et al. | DrAC | — |
| Expected benefit | Lower-variance, unbiased verification credit; correct VoI sign with the mean-only baseline | Correct VoI sign in expectation | Verify before high-risk, irreversible writes | Renamed-tool and unseen-workflow binding | Dropped second repair step; post-evidence binding |
| Failure mode | (i) With group std it over-verifies (AD-T0b). (ii) It breaks the within-group pairing of the shared world, so it can lose to raw/none when repair rarely succeeds. (iii) Needs a known prior; misspecification biases it. | World-luck variance; it changes every task's weighting, not only the verification decisions | Over-conservative | Collapses to the wrong action | see registration |
| Falsification experiment | AD-D1: awrbn vs nostd vs grpo at c = 0.05 (VSC, J_c, verify-after-write); c = 0.30 intervention (awrb should over-verify, awrbn should not) | AD-D1: nostd vs grpo | Not scheduled (budget) | Not scheduled (budget) | C5G_DEV.md kill criteria |

## 3. Rejected or deferred ideas

| Idea | Why |
|---|---|
| Explicit VoI bonus in the reward | Double-counts information already in the policy gradient and changes the fixed point (mandate Prop. 1). It is reward shaping, not an estimator. |
| Lagrangian read budget (constrained GRPO) | Published (Constrained GRPO; Safety as a Budget). Under group std the multiplier cancels (Safety as a Budget, as transcribed in NOVELTY_AUDIT.md §2.4), so it inherits the same pathology. |
| World-decoupled groups (an independent world per rollout) | AD-T0: still biased under group std; it needs no-std, and is then dominated by B or A. |
| Batch-std scaling | AD-T0b: unreliable with 4 groups per batch. |
| Matched-world CRN with forked continuations | Mandate §6: equals GRPO in expectation; k× rollouts. Its idealised Q (AD-T0b's q/group) is correct but costs forks; AWRB is the zero-fork part of it. |
