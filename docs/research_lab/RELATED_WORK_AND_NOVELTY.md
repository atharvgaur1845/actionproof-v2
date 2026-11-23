# Related work and novelty audit (research lab, 2026-11-24)

This file extends `docs/research/algorithm_discovery_v2/LITERATURE_AND_NOVELTY_AUDIT.md` (N1–N9, E1–E11, P1–P6,
C1–C2, not repeated) and `docs/research/NOVELTY_AUDIT.md`.

**Read labels:**
- **[eq]**: the equations were read this session, from the arXiv HTML or the PDF text;
- **[abs]**: the abstract only;
- **[std]**: a standard reference cited from knowledge and not re-read this session.

Searches were run on 2026-11-24 and cover work available up to that date. Absence from a search is weak evidence.

## 1. Primary sources whose equations were checked this session

| # | Work | Read | Equation or statement checked | Bearing on our claims |
|---|---|---|---|---|
| R1 | Bereket and Leskovec, *Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes*, arXiv 2508.11800 | [eq] | Â^STD = (r − mean)/(std + ε); E[Â^NO-STD] = ((G − 1)/G)·A; E[Â^STD] ≈ p(r(p̂,1) − μ₁)/(σ₁ + ε) + (1 − p)(r(p̂,0) − μ₀)/(σ₀ + ε). There is no formal theorem (the authors say a formal characterisation is out of scope). PPO, RLOO and no-std GRPO are calibrated | **The 1/σ-per-outcome weighting is theirs.** Our Theorem 1 is the exact finite-G version for a two-valued return, with a *hidden state* in place of their binary outcome, and its consequence for a costed information action. They do not treat costs, information-gathering actions or a state shared within the group |
| R2 | Bay and Yearick, *GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number: The Group-Standard-Deviation Identity*, arXiv 2607.00152 | [eq] | Binary rewards, k correct of G: GRPO update g = σ·(s̄₊ − s̄₋), σ = √(k(G − k))/G; Dr. GRPO's σ²·Δs. Non-binary rewards, costs and shared latents are explicitly set aside | **Theorem 1 is their identity extended** to any two-valued return R = α + β·r (the update size is the std of the *indicator*, so it is free of β) and mixed over hidden states. That mixture is the step that produces the stake-free vote |
| R3 | GDPO, arXiv 2601.05242 | [eq] | Eq. 4: per-reward A_k = (r_k − mean_k)/std_k within the group; Eq. 5: sum; Eq. 6: batch normalisation. No ε in Eq. 4; cost penalties are not analysed | Our prediction (v2) that a cost component becomes an O(1) anti-read vote in every group is derived from Eq. 4. TB1: GDPO has the worst wrong-decision rate, 370/495 = 0.747 |
| R4 | Schulman et al., *Trust Region Policy Optimization*, arXiv 1502.05477, §5.2 "vine" | [eq] (PDF text) | "For each of these states s_n, we perform multiple actions … and perform a rollout after each action, using common random numbers (CRN) to reduce the variance"; credits Lagoudakis and Parr 2003 and Gabillon et al. 2013, and Ng and Jordan 2000 for CRN in RL | **The paired-fork + CRN estimator at the heart of CDVO is TRPO's vine** |
| R5 | Dimitrakakis and Lagoudakis, *Rollout Sampling Approximate Policy Iteration*, Machine Learning 72 (2008), arXiv 0805.2027 | [eq] (PDF text) | Rollout allocation over states as a bandit problem (counting, UCB, successive elimination), with a Hoeffding test to stop sampling states once a dominating action is identified; about 10× fewer rollouts | **CDVO's "branch where Δ is uncertain" is RSPI's allocation rule** |
| R6 | Li et al., *Counterfactual Rollout Replay: Forkable Environments as Free Process Rewards for Software Engineering Agents*, arXiv 2609.33875 (NeurIPS 2026) | [abs] | Selects a small set of decision points in an on-policy trajectory, forks their states, samples an alternative action, rolls forward "under the policy", and replaces the step's advantage with the difference between realised and counterfactual terminal return. Equal wall-clock comparison: 41.7% vs 36.7% on SWE-bench Verified. Limitation: needs "affordable, reliable state restoration" | **The closest prior method to CDVO for LLM agents.** It shares forked on-policy counterfactual credit at selected agent decisions. It differs by contrasting the realised action with one sampled alternative (CDVO uses all actions of the verify decision and CRN), and it does not target costed verification. Unknown: whether it helps when the action's value depends on a hidden state |
| R7 | Khomich, Hermansson, Hakimi, *EPIG-Tree: Compute-Optimal Branching for Gradient-Efficient RL*, arXiv 2609.20004 | [abs] | Branch placement that most reduces policy-gradient uncertainty per unit compute: n_e ∝ w_e·‖∇log π(a_e\|h_e)‖·σ_e/√c_e. Wins in 9 of 13 control environments and Wordle (0.850 vs 0.790) | **Prior art for CDVO's adaptive branching**, and a better-founded allocation law than the sign-uncertainty rule we test |
| R8 | Liu et al., *Agents' Last Doubt: Knowing When to Act and When to Step Back in Tool-Using Agents*, arXiv 2610.24194 | [abs] | Counterfactual dataset: 3,115 action outcomes at 1,036 ToolSandbox decision points, each legal action executed and scored on task completion, interaction cost and irreversible errors; trains a controller (SURE-Act) for a frozen backbone | **Closest on the decision itself** (act / gather information / escalate, costed). It differs by being offline supervised selection for a frozen agent, not RL credit |
| R9 | Dong et al., *Agentic Reinforced Policy Optimization (ARPO)*, arXiv 2507.19849 (ICLR 2026) | [abs] | Entropy-based adaptive branching after tool calls; soft advantage estimation for shared prefixes | Adaptive branching at tool-call steps in agentic RL |
| R10 | Baisero and Amato, *Unbiased Asymmetric Reinforcement Learning under Partial Observability*, AAMAS 2022, arXiv 2105.11674 | [abs] | A common state-critic variant of asymmetric actor-critic is biased under partial observability. They propose an unbiased alternative. Per the paper body (from knowledge, [std]): the history-state critic Q(h, s, a) | Fork targets from the actual simulator state are samples of Q(h, s, a), which is why CDVO is unbiased (CDVO_THEORY §2) |

## 2. Standard references (not re-read this session; cite after checking)

| # | Work | Relevance |
|---|---|---|
| S1 | Liu et al., Dr. GRPO, arXiv 2503.20783 | std removal; our `nostd` baseline |
| S2 | Kool, van Hoof and Welling 2019; Ahmadian et al., arXiv 2402.14740 (RLOO) | leave-one-out baseline |
| S3 | Schulman et al., PPO, arXiv 1707.06347; GAE, arXiv 1506.02438 | critic baseline; `ppo`, `critic_v` |
| S4 | Foerster et al., COMA, AAAI 2018, arXiv 1705.08926 | counterfactual baseline (CDVO_THEORY §7) |
| S5 | Allen et al., Mean Actor-Critic, arXiv 1709.00503; Ciosek and Whiteson, EPG, arXiv 1706.05374 | all-action gradient |
| S6 | Kazemnejad et al., VinePPO, arXiv 2410.01679 | Monte Carlo value from re-sampled continuations |
| S7 | Kaelbling, Littman and Cassandra 1998 (POMDPs); Silver and Veness 2010 (POMCP) | belief-state RL; root sampling |
| S8 | Howard 1966 (information value theory); Krause and Guestrin 2005/2009 (near-optimal VOI) | value of information |
| S9 | Nam, Fleming and Brunskill 2021 (ACNO-MDP); Krale et al. 2023 (Act-Then-Measure) | costly observation in RL |
| S10 | Achiam et al. 2017 (CPO); Tessler et al. 2019 (RCPO); Altman 1999 | constrained RL; Lagrangian cost |
| S11 | Ng and Jordan 2000 (PEGASUS) | common random numbers in RL |
| S12 | Dudík, Langford and Li 2011; Jiang and Li 2016 | counterfactual evaluation (doubly robust OPE) |
| S13 | Sutton et al. 1999 (policy-gradient theorem); Kakade and Langford 2002 (performance difference) | CDVO_THEORY §2, §4 |

Paired interventions inside a group (same fault schedule) are covered by Sakib, arXiv 2609.24144 (v2 N8, [abs]).

## 3. Searches run on 2026-11-24

1. "counterfactual rollout branching verify vs continue action value LLM tool agent reinforcement learning 2026
   paired forks common random numbers" (extended). It found R6, R7 and R8, plus *The Replay Gap* (arXiv
   2608.08239), *Don't Solve, Just Compare* (arXiv 2608.21027) and BiPACE (v2 E10). None uses CRN by name; all fork a
   restored state.
2. "ARPO … adaptive rollout branching after tool call" found R9 and its follow-up AEPO (arXiv 2510.14545).
3. "Dimitrakakis Lagoudakis rollout sampling approximate policy iteration" found R5.
4. The v2 searches (four, listed in the v2 audit) found no paper combining group std normalisation with a hidden
   state shared within the group and costed verification.

## 4. Prior-art comparison for each candidate contribution

| Candidate | Closest method | Shared | Different | Unknown | Verdict |
|---|---|---|---|---|---|
| **C-A. Stake-free vote (Theorem 1) and the normaliser condition** | R1, R2; v2 N8 (paired rollouts) | 1/σ weighting (R1); update = indicator std (R2) | Return of any two-valued magnitude; mixture over a hidden state shared within the group; consequence: GRPO's expected update is E_z[sign β_z], not E_z[β_z], so it verifies iff p·s > 1/2, cost-blind; exact finite-G statement; the normaliser-independence condition and its predictions for batch std, GDPO, fork/tree-node std and Lagrangian costs | Whether a 2026 paper we did not find states the same mixture result; whether it is observed under confirmatory LLM conditions (E1 never ran) | **Defensible as an analytical contribution of modest size**: a corollary that prior work did not state, with exact testbed confirmation (494/495 cells predicted). It should be presented as building directly on R1 and R2, not as a new mechanism in kind |
| C-B. AWRB (first stage) | Input-dependent baselines; SAMPLR | Integrate the return over observation-consistent worlds | Exact replay; prior weights | — | Not superior (v2 audit, negative) |
| C-C. igc / igc_v1 | Per-step credit; exact local values | Per-decision credit | −c is exact on provably uninformative first reads (support only) | LLM effect (D3 tasks 1216/1217 not finished) | Weak to moderate; development only; speed, not endpoint |
| **C-D. CDVO (paired forks, on-policy continuation, CRN, verify decisions)** | R4 (TRPO vine + CRN), R6 (CRR), S4 (COMA), S6 (VinePPO) | All elements: forced-action forks from a restored state, on-policy continuations, CRN, counterfactual advantage | Only the choice of decision points (verification after aliased acknowledgements) and the all-action contrast at them | Whether it pays off at matched compute in costed verification: tested in LAB1 | **Not novel.** It fails the first go/no-go item before any result. LAB1 tests it as a known family |
| C-E. Adaptive branch allocation | R5 (RSPI), R7 (EPIG-Tree), R9 (ARPO) | Allocate rollouts to uncertain decision points | Sign-uncertainty of Δ per observable context | Whether it beats random allocation here: LAB1 descriptive | **Not novel** |
| C-F. Learned history-conditioned critic from fork means | Mean actor-critic (S5), asymmetric AC (R10) | Learned Q over histories | Trained on fork targets | — | Not novel |

## 5. Conclusion of the audit

- **No algorithmic candidate survives novelty review.** CDVO is TRPO's vine estimator with COMA's counterfactual
  advantage, applied at verification decisions. CRR (NeurIPS 2026) already applies forked on-policy counterfactual
  credit to LLM agents. Adaptive allocation is RSPI and EPIG-Tree.
- **The surviving contribution is analytical and diagnostic:** C-A, plus the negative results (AWRB, forks at
  matched budget, igc's endpoint) that show where credit-assignment fixes do and do not help in costed verification.
- Under the author's rule, the paper should be the "strongest possible mechanistic paper". It should not claim a
  new method.
