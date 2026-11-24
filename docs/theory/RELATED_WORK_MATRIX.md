# Related work and novelty matrix (2026-11-24)

**Read labels:**
- **[eq]**: equations or formal statements read in this or the previous lab session (arXiv HTML/PDF);
- **[abs]**: abstract or landing page only;
- **[std]**: standard reference from knowledge, not re-read.

Metadata (title, authors, date) was checked on the arXiv abstract page this session where marked ✓. This file
extends `docs/research_lab/RELATED_WORK_AND_NOVELTY.md` and supersedes it for the novelty verdict.

## 1. Matrix

Column key: **Shared** = hidden state shared across a group's trajectories; **Info** = information acquisition is an
explicit action; **Cost** = action cost or consequence severity is modelled.

| Work (✓ = metadata verified this session) | Read | Result | Setting | Shared | Info | Cost | Overlap with ActionProof | What ActionProof adds |
|---|---|---|---|---|---|---|---|---|
| Liu, Chen, Li, Qi, Pang, Du, Lee, Lin, *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO), arXiv 2503.20783 ✓ (v2 Oct 2025; no venue listed) | [std] | std term causes question-level difficulty bias; remove std and length normalisation | math reasoning | no | no | no | the `nostd` arm is their advantage | the bias becomes a decision error when the "question" is (prompt, hidden state) |
| Bereket, Leskovec, *Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes*, arXiv 2508.11800 ✓ | [eq] | per-outcome 1/σ weighting; GRPO miscalibrated, PPO/RLOO/no-std calibrated; no formal theorem | binary stochastic outcomes | no | no | no | the 1/σ mechanism | exact finite-G results; hidden state shared within the group; costed information action |
| Davis, Recht, *What is the objective of reasoning with reinforcement learning?*, arXiv 2510.13651 ✓ | [eq] (HTML read) | GRPO ≈ gradient ascent on E_x[h(p_x)], h ≈ arcsin√t per prompt; binary rewards; approximate | binary correctness | no | no | no | per-group monotone transform | Thm 3.1 is the per-(prompt, state) analogue for **any** reward and policy (exact, ε → 0), and it is what makes stakes across hidden states invisible |
| Bay, Yearick, *GRPO, Dr. GRPO, and DAPO Are Three Operations on One Number*, arXiv 2607.00152 ✓ (30 Jun 2026) | [eq] (prev. session) | binary rewards: GRPO update = σ·Δs̄, Dr. GRPO σ²·Δs̄, DAPO drops σ = 0; non-binary, costs and latents set aside | binary rewards | no | no | no | Cor. 3.2's per-state step is their identity with a two-valued return | the mixture over hidden states, the vote, and everything in Props 2.1–3.5 |
| Wang, *The Dark Room in the Reward Channel*, arXiv 2607.21273 ✓ (Jul 2026) | [eq] (Prop. 1 read) | in all-fail groups the z-scored advantage is invariant to the shaping coefficient λ | GRPO agents, ALFWorld, single seed | no ("~10% share initial state" only) | no | shaping coefficient | **the closest statement of scale invariance in homogeneous groups** | invariance per hidden state for any return (not only all-fail); its decision consequence for verification; group-law theory |
| Zhu, Chen, Yu, Zhao, Jia, *Stratified GRPO*, arXiv 2510.06214 ✓ | [abs] | cross-stratum bias of global baselines; normalise within structural strata | search agents | no (strata = trajectory structure) | search calls | no | normalising within subgroups | opposite direction: our strata (hidden states) are what *causes* the bias under std. Note: SAN's within-stratum std would also make stakes per stratum invisible (prediction) |
| Feng, Xue, Liu, An, *Group-in-Group Policy Optimization* (GiGPO), arXiv 2505.10978, NeurIPS 2025 ✓ | [abs] | step-level groups by recurring anchor states, plus episode-level groups | ALFWorld, WebShop | episode groups share task and start state | no | no | groups that share environment state | identifies a failure mode of state-sharing groups under std normalisation |
| Wang et al. (… Manling Li), *RAGEN-2: Reasoning Collapse in Agentic RL*, arXiv 2604.06268, ICML 2026 | [abs] | low within-prompt reward variance lets regularisers dominate; SNR-aware prompt filtering | agentic RL | not analysed | no | no | variance-driven collapse | different mechanism (regularisation vs signal); ours is a sign error, not weak signal |
| Sakib, *Luck Is Not Skill: When Do Paired Rollouts Help Group-Relative RL of LLM Agents?*, arXiv 2609.24144 ✓ (Sep 2026) | [abs] | sharing an event-keyed noise schedule within a group removes contrast variance; gradient variance need not fall | tool-use agent with faults | **yes, by design** (paired noise) | no | no | **closest on group construction** | if paired groups are std-normalised, Thm 3.1 predicts the paired component is voted, not weighed (untested on their setup) |
| Ding, Tomlin, Durrett, *Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents*, arXiv 2602.16699 ✓ (NYU) | [eq] (RL section read) | GRPO baseline "collapses to a static policy which always triggers unit tests" across cost ratios; authors attribute it to RL not learning the prior | costed exploration, Qwen3-8B | **not stated** | yes | yes (multiplicative discounts) | closest on the decision | cost-insensitive collapse **consistent with** Thm 3.1 if their groups share the latent; group construction unreported, so **unverified** |
| Mroueh, *RLVR: GRPO's Effective Loss, Dynamics, and Success Amplification*, arXiv 2503.06639 ✓ | [abs] | GRPO as adaptive weighted contrastive loss; fixed-point recursion of success probability | binary verifiable rewards | no | no | no | effective-objective analysis | — |
| Liu et al. (13 authors), *GDPO: Group reward-Decoupled Normalization Policy Optimization*, arXiv 2601.05242 ✓ (NVIDIA tech report) | [eq] (prev.) | per-reward group normalisation, sum, batch norm | multi-reward | no | no | components | our Prop. 4.3 is about their per-component normalisation | a separate cost component becomes a full anti-read vote (never verifies) |
| Yu, Zhang, Zhu, Yuan, Zuo, et al. (35 authors), *DAPO*, arXiv 2503.14476 ✓ | [std] | dynamic sampling (drop σ = 0 groups), clip-higher, token-level loss | math | no | no | no | TRL's `dapo` loss used in ActionProof | Prop. 4.4: filtering cannot change the vote's sign |
| Ahmadian, Cremer, Gallé, Fadaee, Kreutzer, Pietquin, Üstün, Hooker, *Back to Basics* (RLOO), arXiv 2402.14740 ✓ | [std] | RLOO for RLHF | RLHF | no | no | no | `rloo` baseline | Prop. 2.2 |
| Kool, van Hoof, Welling, *Buy 4 REINFORCE samples, get a baseline for free!* (2019, ICLR DeepRLStructPred workshop) | [std] | leave-one-out baseline | — | — | — | — | — | — |
| Schulman et al., PPO (1707.06347); TRPO (ICML 2015) | [std]/[eq] | clipping; vine + CRN | — | — | — | — | baselines | — |
| Howard 1966, *Information Value Theory*, IEEE Trans. SSC | [std] | VOI | decision theory | — | yes | yes | VOI definition | — |
| Kaelbling, Littman, Cassandra 1998, *Planning and acting in partially observable stochastic domains*, AIJ | [std] | POMDPs | — | — | — | — | aliasing (A1) | — |
| Nam, Fleming, Brunskill 2021 (ACNO-MDP), NeurIPS; Krale, Simão, Jansen 2023 (Act-Then-Measure), AAMAS | [std] | costly observation in RL | — | no | yes | yes | costed information acquisition | they do not study group-normalised gradients |
| Ng, Jordan 2000, PEGASUS, UAI | [std] | common random numbers | — | yes (CRN) | — | — | correlated rollouts | — |
| Tamar, Glassner, Mannor 2015, *Optimizing the CVaR via sampling*, AAAI | [std] | risk-sensitive PG | — | — | — | yes | the `cvar` reward option | not used in claims |

**Searches run this session (2026-11-24).**
- "GRPO agent environment stochasticity same seed rollouts group hidden state advantage normalization bias"
  (extended). It found Bereket and Leskovec, Dark Room, Stratified GRPO, RAGEN-2, AlphaBet and a seed-pinning bug
  report. None addresses a hidden state shared within the group together with costed information actions.
- Searches for CTA, GiGPO, "ERPO" (no matching paper found under that name) and Mroueh.
- The previous session's four searches.

Absence from search is weak evidence.

## 2. Mandatory novelty test

1. **Is the theorem a specialisation of an existing GRPO std result?**
   - *Partly yes.* The per-state step (one shared-state group) is Bay and Yearick's identity with a two-valued return,
     and Theorem 3.1's invariance is the affine invariance of z-scores noted for all-fail groups by Wang (Dark Room
     Prop. 1).
   - What is not in prior work: treating the hidden environment state as part of the group key, and the decision
     consequence. Severity across states is provably invisible, which yields the vote and its exact threshold scope.
2. **Is the hidden-state coupling itself the new mathematical contribution?**
   - The coupling is a modelling observation (group key = prompt × hidden state), not deep mathematics.
   - The new mathematics lies around it:
     - the π-free Bernstein representation (Prop. 2.1);
     - law-invariance of centred estimators;
     - **endogenous homogeneity** (Cor. 3.3: decorrelating states cannot remove the vote near always-verify, and the
       regret is Θ(1/G));
     - the critical-correlation results and the insufficiency of pairwise correlation (Props 3.4–3.5).
3. **Is the VOI interpretation new?**
   - No. Dr. GRPO ∝ ΔQ is the policy-gradient theorem with a leave-in baseline.
   - Calling it VOI is correct only with Bayes-optimal continuation, which does not hold in LLM training.
4. **Does it apply to realistic samplers?**
   - Yes for ActionProof (verified in code and in 800/800 logged groups).
   - Plausibly for task-instance agent benchmarks where the environment snapshot is fixed per task, but that is not
     verified for any external system.
   - Paired-noise designs (Sakib) create it deliberately.
5. **Does the language-agent evidence validate the mechanism?**
   - **Partly, development-level**:
     - std-normalised arms follow the early vote (5/5 runs);
     - per-state cost invariance holds on real rollouts;
     - group sharing is confirmed.
   - Dr. GRPO's rise in reading is **not** predicted by the warm-start ΔQ.
   - Confirmatory evidence awaits AD-C1 (predictions frozen in `docs/research/AD_C1_THEORY_PREDICTION.md`).
6. **Is there a genuinely new corollary in the partial-correlation extension?**
   - Yes. These are, as far as the searches show, not stated before:
     - Cor. 3.3 (law-free corner);
     - Θ(1/G) interior regret under independent states (numerical);
     - the family-dependent critical correlation.
   - They are modest.

## 3. Verdict

**Incremental but defensible.**
- The core mechanism (std normalisation removes scale within a group) is known in several forms.
- The defensible contribution is identifying that agentic groups are keyed by hidden environment state, and deriving
  exactly what that does to a costed information action:
  - stake-blindness;
  - the precise scope of p·s > 1/2;
  - why independent resampling is not a cure;
  - which estimators are and are not affected.
- This is supported by a registered held-out testbed and development-level LLM diagnostics.

It is not "clearly novel". It is more than "a known result applied to a new domain" because of Cor. 3.3 and the
group-law results.
