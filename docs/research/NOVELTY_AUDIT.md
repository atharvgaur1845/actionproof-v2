# ActionProof novelty audit: candidates A, B and C

Compiled 2026-11-21. Purpose: decide how much of each Stage 4 candidate is already done in prior work, before any of them is built or claimed. This file complements `docs/stage3/LITERATURE.md` (cited below as [LIT xx]); it does not repeat those annotations.

Candidates under audit:

- **A. Counterfactual state-forked branches.** At a safety-critical state, fork the exact simulator state, sample K branches, group them by the first decisive action class, compute A(s,a) = Q(s,a) - sum_a' q(a'|s) Q(s,a'), and apply it to the decisive-action tokens, plus an episode-level advantage.
- **B. Recovery-aware constrained GRPO.** Separate costs (irreversible action, false completion, unnecessary action) with per-component group normalisation.
- **C. Adaptive recovery-frontier scheduling.** Choose start states by failure probability, severity, uncertainty and learning progress.

---

## 0. Method and verification rules

- Every paper in sections 2 to 6 and in the table was fetched in this session. Abstract pages came from arxiv.org/abs. Methods were read from the arXiv HTML full text (arxiv.org/html) where the entry says "HTML read". Pre-2023 papers without arXiv HTML were read on ar5iv (TRPO, COMA, SAMPLR). Venues were checked on ACL Anthology, PMLR, icml.cc or neurips.cc pages where stated. No PDF was downloaded.
- Method details were extracted from the HTML text by a summarising fetch tool. The equations below are transcriptions of those extractions. Before an equation is quoted in a paper, re-open the cited HTML section and copy it from there.
- Venue labels: "verified at X" means a proceedings, anthology or event page was fetched. "Per arXiv" means the venue appears only in the arXiv comments field. "arXiv preprint" means the fetched page showed no venue.
- [LIT xx] entries are verified and annotated in `docs/stage3/LITERATURE.md`. They appear here only where they bear on A, B or C. Six were re-fetched in this session for method detail (Constrained GRPO, GiGPO, VinePPO, COMA, R3L, LIMBO).
- Anything seen only as a search snippet is listed under "UNVERIFIED (not usable)". One fetched paper was withdrawn by arXiv and is listed there too.
- Dates are arXiv v1 dates unless stated otherwise.

## 1. ActionProof reference setting

- Policy: 3.35B tool-using LM, LoRA, GRPO through TRL AgenticGrpo. Sequence-level group-relative advantage, G = 8 rollouts per prompt.
- Faults: phantom ack (write not committed, ack identical to success), partial commit at ratio q, lost ack, duplicate delivery, never-commit. The ack can be byte-identical whether or not the write committed.
- Information split: the verifier and reward see hidden authoritative state; the policy sees only tool outputs.
- Stage 3: recovery-start curriculum. Replayed scripted prefixes end in faulted states; the GRPO update is unchanged. Half of the Stage 3 training pool was recovery starts (`docs/stage3/RESEARCH_REPORT.md`).

**The structural property used throughout this audit.** In ActionProof the fault is an observation-equivalent hidden variable: two histories can be identical while the hidden commit status differs. Call this OEHF (observation-equivalent hidden fault). Any training step that selects, weights or scores data using the hidden state, while the policy conditions only on the history, changes the effective prior p(fault | history) that the policy is trained under. Almost none of the prior work below has this property. It is the main source of whatever novelty remains.

---

## 2. Required papers (items 1 to 7)

### 2.1 Fission-GRPO
- **Title:** Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors
- **Authors:** Zhiwei Zhang, Fei Zhao, Rui Wang, Zezhong Wang, Bin Liang, Jiakang Wang, Yao Hu, Shaosheng Cao, Kam-Fai Wong
- **Venue:** Proceedings of the 64th Annual Meeting of the ACL (Volume 1: Long Papers), pp. 40477-40491, July 2026 (verified at ACL Anthology, 2026.acl-long.1880). arXiv v1 2026-01-22, v2 2026-04-20. Read: abstract, HTML v1 and v2, Anthology page.
- **Problem:** after a tool-call error, small models repeat invalid calls. Standard RL turns failures into sparse negative reward, and static error-correction data drifts away from the policy's current failure modes.
- **Update rule:** Stage 1 is standard GRPO. Failed rollouts are intercepted. Each becomes a corrective context x_corr = [x; tau_err; f]. For format errors f is a deterministic message. For semantic errors f = S_phi(x, tau_err, tau_gt), written by a fine-tuned Qwen3-32B "Error Simulator" (about 2K training examples) that sees the reference call but is trained not to reveal it. From x_corr the method samples a "fission group" of G' recovery rollouts. Advantages are standardised within that group, and the same clipped GRPO objective with KL is applied. Corrective contexts are kept in a LIFO buffer, so the newest errors are trained on first.
- **Reward:** (1/3)[w_fmt(t) R_fmt + w_corr(t) R_corr + R_len]. R_corr is function-name match plus token F1 against the reference call. The format weight decays and the correctness weight grows over training.
- **Policy vs evaluator:** the policy sees its own failed call and explicit error text f. The reward and the Error Simulator see the reference call.
- **Failed trajectories, recovery, irreversibility:** the failed trajectory itself becomes the prefix of a new group. Only the text context is reused; no environment state is forked or restored, and side effects of the failed call are neither undone nor modelled.
- **Setup:** BFCL v4 Multi-Turn, TAU-Bench, TAU2-Bench. Qwen3 1.7B, 4B, 8B. Baselines: GRPO, DAPO, Dr.GRPO, AWPO, ToolACE-2-8B, BitAgent-8B. Qwen3-8B overall accuracy 42.75% to 46.75%; error-recovery rate +5.7 points.
- **Differs from ActionProof:** errors are explicit and turned into diagnostic text. ActionProof's faults are silent (byte-identical ack), so from the policy's side there is no error event to fission on. Reward is call matching, not authoritative state.
- **Redundant if:** ActionProof claims that restarting GRPO groups from failure contexts teaches recovery. The Stage 3 recovery-start curriculum is a scripted-prefix variant of this idea, and also of R3L [LIT H6], failure-prefix conditioning [LIT C7] and ReflectPO (section 3). Any candidate that writes LLM-generated failure feedback into the context is Fission-GRPO or ReGRPO.

### 2.2 ReGRPO
- **Title:** ReGRPO: Reflection-Augmented Policy Optimization for Tool-Using Agents
- **Authors:** Binjie Zhang, Mike Zheng Shou
- **Venue:** arXiv preprint, v1 2026-06-30. Read: abstract, HTML.
- **Problem:** tool-augmented VLMs are fragile. SFT on successful trajectories gives little recovery signal, and a sparse trajectory reward does not say which step failed or how to fix it.
- **Data:** starting from a ground-truth step, the method perturbs the tool choice or arguments ("near-miss" actions) and executes them in the sandbox to get real failure observations. It then builds Reflection-of-Thought triplets (ErrorType, Evidence, FixPlan) plus a corrected action for an SFT warm start.
- **Update rule:** GRPO-style with K = 5 candidates per local segment. A segment is either one-shot (a0, o0) or reflection-triggered (a0, o0, z, a1, o1). Baseline = group mean. Reward R = lambda_exec 1{success} - eta C(tau) + lambda_val V(x, tau), with C proportional to reflection token length (0 for one-shot success). Defaults: 1.0, 0.1, 0.3. A KL term is added.
- **Policy vs evaluator:** the policy sees failed action outputs (empty OCR, API errors, irrelevant crops) and its own reflection. The reward uses environment success and optional metadata-based verifier scores.
- **Failed trajectories, recovery, irreversibility:** RL does not restart from failure states; it optimises short local segments. No side effects are discussed.
- **Setup:** GTA (229 tasks), GAIA (446 tasks). Qwen2-VL-7B, LoRA rank 32. Baselines: MAT-Agent, SPORT, HF Agents.
- **Differs from ActionProof:** failures are visible and local, with a single retry, in vision tools that have no persistent writes.
- **Redundant if:** ActionProof adds a reflection or diagnosis step with a reflection-length cost and presents it as its recovery mechanism, or claims that learning from executed near-miss failures is new.

### 2.3 Constrained Group Relative Policy Optimization [LIT D7, HTML v4 re-read]
- **Authors:** Roger Girgis, Rodrigue de Schaetzen, Luke Rowe, Azalee Robitaille, Christopher Pal, Liam Paull
- **Venue:** arXiv preprint, v1 2026-02-05, v4 2026-09-02.
- **Problem:** how to add Lagrangian constraints to GRPO.
- **Update rule:** max_pi min_{lambda >= 0} J_R - sum_k lambda_k (J_Ck - d_k), with indicator costs C_k = 1[constraint k violated]. Each component is standardised within the group first: Z_R,i = (R_i - mean R)/std_R and Z_Ck,i = (C_k,i - mean C_k)/std_Ck. They are then combined: A_i = lambda_R Z_R,i - sum_k lambda_k Z_Ck,i. Multipliers are a softmax over logits and are updated by the dual gradient -(J_Ck - d_k). Theorem 4.1 shows that "scalarise then standardise" couples each multiplier to all others through the shared denominator.
- **Policy vs evaluator:** single-trajectory tasks; costs are indicator functions on outputs.
- **Failed trajectories, recovery, irreversibility:** none. No multi-turn or tool-use experiments.
- **Setup:** gridworld (2-layer MLP), NAVSIM-v2 (Qwen2.5-VL 3B), GSM8K (Qwen2.5-Instruct 1.5B, 3B, 7B). Baselines: scalarised GRPO, PPO, PPO-Lagrangian, GDPO.
- **Differs from ActionProof:** no agent, no hidden state, no recovery states.
- **Redundant if:** candidate B is this method with K = 3 costs {irreversible, false completion, unnecessary action}. That is exactly its scalarised-advantage form.

### 2.4 Safety as a Budget
- **Title:** Safety as a Budget: Calibrating the Control Channel of Constrained GRPO for Vision-Language Models
- **Author:** Bo Han
- **Venue:** arXiv preprint (comments: 10 pages, 5 figures, 5 tables), v1 2026-11-12. Read: abstract, HTML.
- **Problem:** constrained GRPO for VLM safety, with budgets on the harmful-assistance rate and on the over-refusal rate.
- **Update rule:** r = r0 - lambda_h p~_h 1[x in H] - lambda_r p~_r 1[x in B]. Dual update lambda_k <- clip(lambda_k + eta (g_k - eps_k), 0, lambda_max). The paper shows that standard group std normalisation cancels the multiplier: on harmful prompts A_i = -(1 + lambda_h)(p_h,i - mean p_h)/s, and the (1 + lambda_h) factor divides out. It uses Constrained GRPO's per-objective z-scores, or unscaled advantages as in Dr. GRPO. A fitted isotonic map (about 200 labelled responses) is applied to the penalty and to the dual update (the "control channel"), while raw scorer probabilities stay in r0 (the "learning channel"). The map fixes the scorer's level bias and keeps its ranking.
- **Policy vs evaluator:** the policy sees image and text. The scorer sees request and response text.
- **Failed trajectories, recovery, irreversibility:** none; single-turn.
- **Setup:** Qwen3-VL-2B with LoRA. MM-SafetyBench, FigStep, JailbreakV-28K, MOSSBench contrast set, HADES (held out), MMBench. Baselines: GRPO without budget, Constrained GRPO, fixed-weight penalty, safety prompts. Held-out harm falls from 8.9% to 4.8%; over-refusal stays at or below 2%.
- **Differs from ActionProof:** single-turn; the cost comes from a learned judge with level bias. ActionProof's costs come from authoritative state, so the calibration problem is absent, but the multiplier-cancellation problem applies unchanged.
- **Redundant if:** B claims that GRPO normalisation defeats Lagrange multipliers, or that two-sided budgets (unsafe action and over-refusal) prevent collapse to always-refuse.

### 2.5 Proof-of-Use
- **Title:** Proof-of-Use: Mitigating tool-call hacking in Deep Research Agents
- **Authors (Anthology):** Shengjie Ma, Chenlong Deng, Jiaxin Mao, Jiadeng Huang, Xuhui Jiang, Teng Wang, Junjie Wu, Changwang Zhang, Jun Wang. The arXiv v3 page lists the same authors without Xuhui Jiang; use the Anthology list.
- **Venue:** Proceedings of EMNLP 2026, pp. 37843-37857, October 2026 (verified at ACL Anthology, 2026.emnlp-main.1664). arXiv v1 2025-10-13, v3 2026-10-08. Read: abstract, HTML v3, Anthology page.
- **Problem:** "tool-call hacking". RL-trained deep-research agents earn reward through valid-looking tool calls and correct answers while their reasoning does not depend on what the tools returned (tool overuse and decorative calls).
- **Update rule:** GRPO, 8 rollouts. Reward = (R_cite + R_pt + R_a)/3. R_cite: citation parsing and consistency with cited IDs (+1/-1). R_pt: evidence sensitivity, tested by degrading cited evidence or inserting lures and checking that the helpfulness verdict changes. R_a = alpha_s R_ans + (1 - alpha_s) R_anc mixes answer-support faithfulness (Ragas) and answer correctness (LLM judge); alpha_s decays with an EMA of correctness.
- **Policy vs evaluator:** the policy sees tool outputs with stable paragraph IDs. The reward checks citations against tool outputs, runs perturbation tests and uses an LLM judge.
- **Failed trajectories, recovery, irreversibility:** retrieval tools only. The paper excludes execution tools because their failures are "typically observable".
- **Setup:** Qwen2.5-7B-Instruct; HotpotQA and 2Wiki in distribution; NQ, TriviaQA, MuSiQue, PopQA, Bamboogle, BioASQ, PQArefEval out of distribution. Baselines: Search-o1, Search-R1, R1-Searcher, DeepResearcher, Chain-of-Agents, RAG variants.
- **Differs from ActionProof:** read-only tools with observable content; no hidden state and no writes. The overlap is conceptual: decorative tool calls resemble ActionProof's "acquisition without integration" finding (read-backs that do not change the report).
- **Redundant if:** ActionProof claims as new that outcome RL produces tool calls whose results do not causally drive the decision, or claims evidence-perturbation tests as a new diagnostic. It does not make A, B or C redundant.

### 2.6 Reward Hacking Benchmark
- **Title:** Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use
- **Author:** Kunvar Thaman
- **Venue:** Proceedings of ICML 2026, PMLR 306, pp. 120372-120389 (verified at PMLR and at the icml.cc poster page). arXiv v1 2026-05-03. Read: abstract, HTML, PMLR page.
- **Problem:** how often tool-using agents take shortcuts instead of solving multi-step tasks honestly.
- **Method:** evaluation only. Six exploit categories: leakage or metadata use, tampering, sequence manipulation (skipping upstream work, fabricating downstream artifacts or completion markers, skipping mandated verification steps), proxy or parser gaming, special-casing visible checks, and denial of evaluation.
- **Policy vs evaluator:** the agent sees the task, input files, bash and Python, and a workspace with task-adjacent metadata. The evaluator sees executed commands, file accesses, hashes of protected files, and runs a hidden recomputation on unobserved data.
- **Recovery, irreversibility:** false completion markers and skipped verification are measured; no duplicate-write or rollback scenarios.
- **Setup:** 13 frontier models. Exploit rates range from 0% (Claude Sonnet 4.5) to 13.9% (DeepSeek-R1-Zero, vs 0.6% for DeepSeek-V3). 72% of exploit episodes contain explicit CoT rationale. Environment hardening cut exploits from 6.5% to 0.8% (87.7% relative).
- **Differs from ActionProof:** no training; shortcuts are available to the agent by design, whereas ActionProof's false completions arise because the agent cannot tell commit from non-commit.
- **Redundant if:** ActionProof presents "RL-trained agents skip verification and claim completion" as a discovery. It is already measured here and in Confident Closing [LIT F4].

### 2.7 AgentRobustBench / When Simulation Lies
- **Title on arXiv (current):** AgentRobustBench: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents. **Title on the NeurIPS 2026 and ICML 2026 workshop pages:** When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents (benchmark named RobustBench-TC there). Same paper, arXiv 2605.11928.
- **Authors:** Xiaolin Zhou, Aojie Yuan, Zheng Luo, Zipeng Ling, et al. (15 authors)
- **Venue:** NeurIPS 2026 poster, Evaluations and Datasets Track (per arXiv; verified at neurips.cc poster 139169). Also a poster at the ICML 2026 workshop "Failure Modes in Agentic AI" (verified at icml.cc). arXiv v1 2026-05-12, v3 2026-11-07. Read: abstract, HTML, both event pages.
- **Problem:** benchmarks assume clean inputs and reliable APIs; the paper frames deployment failures as a sim-to-real gap in the tool-use POMDP (observation, action space, reward-relevant, transition).
- **Update rule:** ToolRL-DR keeps the public ToolRL GRPO code and replaces the 4,000 training samples with perturbation-augmented samples, drawn uniformly from 16 statically augmentable types (4 observation, 6 action, 6 reward). Transition perturbations are excluded from training. Reward: tool-name and parameter correctness.
- **Policy vs evaluator:** the policy sees dialogue history, tool registry and tool outputs. The reward checks tool name and parameters.
- **Recovery, irreversibility:** transition faults (timeout, 429, auth, 5xx, malformed, schema drift) are explicit error messages, and the model gets one recovery turn. Silent failures (fabricated observation, silent no-op) are deferred to future work (Appendix N, per HTML). No side effects.
- **Setup:** 21 models, 1.5B to 32B; 199 clean samples from BFCL v3, API-Bank, RoTBench, ToolAlpaca, ToolEyes; 3,721 perturbed. Reward-relevant and transition perturbations reduce accuracy by about 40% and 30%. A 3B ToolRL-DR model keeps about three quarters of clean accuracy and is comparable to open 14B baselines.
- **Differs from ActionProof:** no state reward, no silent faults in training, no curriculum (uniform sampling), mostly single-turn.
- **Redundant if:** ActionProof claims that domain-randomised fault exposure during RL improves robustness. It does not cover silent write faults.

---

## 3. Candidate A: closest prior work (item 8)

### 3.1 Fork-the-state estimators

**TRPO "vine" procedure** (Schulman, Levine, Moritz, Jordan, Abbeel; ICML 2015 per the paper; read on ar5iv). Trunk trajectories are sampled, a "rollout set" of visited states is chosen, the simulator is reset to each state, K actions are sampled there (from the policy or uniformly), and each is rolled out. Common random numbers are used across the K rollouts to reduce the variance of Q differences. For discrete actions the surrogate at state s_n is sum_k pi_theta(a_k|s_n) Q^(s_n, a_k). The paper states the method needs arbitrary state resets. This is candidate A's estimator in 2015 form, without action classes and without language models. https://arxiv.org/abs/1502.05477

**Counterfactual Rollout Replay (CRR)** (Yuanhao Li, Hongbo Wang, Xuhong Chen, Yiming Cao, et al.; NeurIPS 2026 per arXiv; v1 2026-09-27, v4 2026-10-31; HTML v3 read).
- Problem: outcome-only RL for SWE agents gives no intermediate credit.
- Update: decision points are chosen by score(t) = H_t (1 + lambda_type type(a_t)), where H_t is grammar-masked entropy and type weights are 0 for passive reads, 0.25 for tests, 0.75 for shell and 1.0 for edits. At k = 4 selected steps the container is restored and one counterfactual action is drawn from a tempered policy that excludes the factual action, q(a|s_t) proportional to pi(a|s_t)^(1/T_cf) 1[a != a_t]. Advantage at the selected step: A_t = R(tau) - R(tau~_t). Proposition 1: under deterministic forks its expectation is Q(s_t, a_t) - E_{a~ ~ q}[Q(s_t, a~)]. Other steps keep the GRPO outcome advantage R(tau) - mean group R (hybrid, Eq. 3). Clipped surrogate with KL.
- Observes: policy sees repository state and tool outputs; reward is a hidden-test pass/fail.
- Recovery and irreversibility: a three-rerun determinism filter admits only stable tasks; no treatment of irreversible effects.
- Setup: Qwen3-14B, LoRA rank 128, 3,638 SWE-Gym and SWE-rebench tasks. SWE-bench Verified 41.7% vs 36.7% for extended GRPO at equal wall clock (fork overhead included). Baselines include GRPO, SWE-Fixer, RLEF, TSR+GRPO, a rubric PRM. GiGPO and VinePPO are not compared.
- Relation to A: forks, an action-conditioned same-state contrast, step-token application and a GRPO episode term are all here. Selection already favours state-changing actions. https://arxiv.org/abs/2609.33875

**Branching Policy Optimization (BPO)** (Bowei He, Yankai Chen, Xiaokun Zhang, Xue Liu; WAIC Academic 2026 per arXiv; v1 2026-07-15; HTML read).
- Sandbox snap/restore operators. Branch points are the top-M steps by per-step entropy (minimum spacing 64 tokens). At each branch point the state is restored and K-1 siblings are sampled from pi. Local advantage A = G_k - (1/(K-1)) sum_{j != k} G_j (leave-one-out over siblings). It is propagated to earlier steps with weight lambda^(t - t'), lambda = 0.95, and segments are weighted by the number of leaves they support. Snapshot cost 42 ms (WebShop) to 1,920 ms (SWE-bench).
- No action classes, no Q(s,a) framing, no safety or irreversibility treatment; assumes restore preserves the transition distribution.
- Setup: WebShop, ALFWorld, SWE-bench Verified; Qwen2.5-7B, Llama-3.1-8B; baselines SFT, PPO, RLOO, GRPO, VinePPO. +5.7 to +5.9 points over GRPO; 38.7% fewer gradient steps to GRPO's final level. https://arxiv.org/abs/2607.14171

**ReflectPO** (Rumei Li, Xunliang Cai, et al.; arXiv preprint, v1 2026-10-22; HTML read).
- Builds a prefix tree over rollouts. Branch points are nodes where outcomes diverge (variance of branch rewards) or where a trajectory revisits a node. At recovery anchors it saves an environment snapshot and restores it to generate recovery trajectories. A guide model writes hints that are visible during generation and removed from the context during the policy update.
- Credit: A_raw(v, a) = mean terminal reward through child u minus mean through parent v. "Recovery gain" for new actions after an anchor = clip((r_recovery - r_original)/sigma_B, -1, 1).
- Setup: BFCL-v3 multi-turn (stateful APIs) and tau2-bench; Qwen3-4B and 8B; baselines PPO, GRPO, Reinforce++, GiGPO, CRITIC and Fission-GRPO.
- Relation to A and to Stage 3: environment forks at recovery points in stateful tool APIs, with a parent-child return contrast, already compared against Fission-GRPO. https://arxiv.org/abs/2610.30933

**ELPO, Learning from the Irrecoverable** (Qiao Liang, Yuke Zhu, Chao Ge, Lei Yang, et al.; arXiv preprint, v1 2026-02-10; HTML read).
- Binary search over tool-call steps of a failed trajectory locates the first irrecoverable step: fix the prefix (tool outputs replayed), sample suffixes, recurse. Budget is allocated by entropy. Branch-level advantage A_b = (r(c_i) - mean_b)/(std_b + eps) over siblings; trajectory-level GRPO advantage; A_hier = lambda_tree A_b + (1 - lambda_tree) A_t with lambda_tree = 0.5. The lower clip bound is relaxed on the error suffix.
- Setup: math, science QA and code with tool-integrated reasoning; Qwen2.5-7B, Qwen3-4B; baselines include ARPO, AEPO, GiGPO, ToRL, DAPO. https://arxiv.org/abs/2602.09598

**Agents' Last Doubt (SURE-Act)** (Qisai Liu, Qingyue Jiao, Bangwei Guo, Wenxi Dai, et al.; arXiv preprint, v1 2026-10-16; HTML read).
- At 1,036 ToolSandbox decision points the environment is cloned and each meta-action is forced for one turn: Ask, Search, Act-Reversible, Commit-Irreversible. The frozen backbone then completes the episode. Recorded outcomes: success s, interaction cost c, critical error r (an irreversible action without authorisation). Utility y = s - lambda c - eta r. 3,115 counterfactual outcomes.
- A separate controller (Qwen3-1.7B encoder with LoRA) is trained by regression plus a pairwise ranking loss to pick the meta-action. The policy itself is not trained with RL.
- Relation to A: "fork, force each decisive action class including an irreversible class, score by success minus costs" is already done; ActionProof's difference would be using these contrasts as a policy-gradient advantage. https://arxiv.org/abs/2610.24194

Other fork or replay methods verified this session:
- **TriGRPO** (Hindsight Credit Assignment over Trees; Penghong Zhao et al.; v1 2026-11-02): when a rollout reaches a state that a successful trajectory visited with a different action, it spawns a sibling with the banked action (text prefix replay). Node advantage A = R(tau_j) - R(tau_i). States that irreversible domains are hard for the method. https://arxiv.org/abs/2611.02706
- **Verified Critical Step Optimization (CSO)** (Mukai Li et al.; ACL 2026 Findings per arXiv): PRM-proposed critical steps in failed trajectories; alternatives are executed by the policy and only verified flips become preference data. https://arxiv.org/abs/2602.03412
- **MAGMA-GEN** (Bernat et al.; CoRL 2026 per arXiv): a privileged coach proposes corrections; a correction is kept only if re-execution from the identical state improves the outcome (robot manipulation, supervised). https://arxiv.org/abs/2609.20056
- **BunshinRL** (Yutong Huang et al.; v1 2026-11-07): kernel-level whole-workspace container forking for agentic RL rollouts (systems paper; makes forks cheap). https://arxiv.org/abs/2611.10453
- VinePPO [LIT C6, re-fetched: ICML 2025 per arXiv], Math-Shepherd [LIT B2], AgentPRM [LIT B4].

### 3.2 Action-conditioned and counterfactual baselines

**BiPACE** (Hanyang Wang, Weijieying Ren, Yuxiang Zhang, Ding Cao, et al.; arXiv preprint, v1 2026-06-24; HTML read).
- Problem: step-level group-relative estimators assume compared steps are equivalent. Exact observation hashing (GiGPO) leaves many singleton groups, and within-group averaging mixes state value with action-specific credit.
- Update: states are clustered by cosine distance between actor hidden states (layer -8 for 7B, epsilon 0.10). Within a cluster C, actions are keyed by a hash of the parsed action body or of the first 8 response tokens. PACE Q-style form: Q^(s, a_i) = mean return of same-key members, V^(s) = mean return of the cluster, A^q = Q^(s, a_i) - V^(s). Diff-peer form: R(i) minus the mean of different-key members. Combined advantage A = A_ep + w A^S with GiGPO's weight, applied to the step's tokens.
- Observes: grouping uses the actor's own hidden states, not the environment state. No environment forking. A bisimulation-based bias bound (2 L epsilon) is given; partial observability is not treated beyond that.
- Setup: ALFWorld, WebShop, TextCraft; Qwen2.5-1.5B and 7B; baselines GRPO, PPO, GiGPO, HGPO. ALFWorld 7B: 97.1 +/- 0.9 vs GiGPO 90.8.
- Relation to A: this is candidate A's estimator (action-key mean minus state mean, added to the episode advantage) without forks. https://arxiv.org/abs/2606.25556

**RTMC** (Tao Wang, Suhang Zheng, Xiaoxiao Xu; arXiv preprint, v1 2026-04-13; HTML read). Matches states across a GRPO group via state-action signatures (view bucketing, content hashing, action classification into tool categories such as view, search, modify, execute, test, finish). First-visit Monte Carlo: Q^(s,a) = mean return of visits, A^(s,a) = Q^(s,a) - V^(s), broadcast to the step's tokens, with prior smoothing of V for rare nodes. SWE-bench Verified with Qwen3-Coder-30B-A3B: +3.2 points over GRPO in the abstract. Relation to A: action-class Q minus V on tool categories, critic-free, no forks. https://arxiv.org/abs/2604.11037

**COMA** [LIT C2, re-read on ar5iv]. A^a(s, tau, u) = Q(s, tau, u) - sum_{u'^a} pi^a(u'^a | tau^a) Q(s, tau, (u^-a, u'^a)). The critic sees global state; each actor sees only its own action-observation history. This is candidate A's baseline with q = pi, and it has the same information split as ActionProof (privileged critic, history-conditioned actor).

**GiGPO** [LIT A9, re-read HTML]. Anchor groups are exact repeats of environment states across a group, found with a hash map. A^S = (R_t - mean R_t over the anchor group)/F_norm with F_norm = std or 1, and A = A^E + omega A^S on the step's tokens. Only naturally repeated states are used; no forking.

Classic action-dependent baselines:
- **Wu et al. 2018**, Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines (ICLR 2018 oral per arXiv): bias-free action-dependent baselines; notes the extension to partially observed and multi-agent tasks. https://arxiv.org/abs/1803.07246
- **Tucker et al. 2018**, The Mirage of Action-Dependent Baselines in RL (ICML per arXiv comment): learned state-action baselines did not reduce variance over state baselines on common benchmarks; earlier reported gains traced to implementation details. https://arxiv.org/abs/1802.10031
- **Q-Prop** (Gu et al.; ICLR 2017 per arXiv): Taylor-expanded off-policy critic as a control variate. https://arxiv.org/abs/1611.02247
- **Counterfactual Credit Assignment** (Mesnard et al. 2020): future-conditional baselines that separate skill from luck, with hindsight restricted to exclude the agent's actions. https://arxiv.org/abs/2011.09464
- **RUDDER** (Arjona-Medina et al. 2018): return decomposition that redistributes delayed reward. https://arxiv.org/abs/1806.07857

### 3.3 Privileged information and aliasing

- **Unbiased Asymmetric RL under Partial Observability** (Baisero, Amato; AAMAS 2022, pp. 44-52, per arXiv). State-only critics used with history-conditioned actors are biased; a history-state critic keeps the policy gradient valid. https://arxiv.org/abs/2105.11674
- **Informed Asymmetric Actor-Critic** (Ebi, Ernst, Bohm, Lambrechts; ICML 2026 per arXiv; v3 2026-06-09): critics can condition on arbitrary state-dependent privileged signals and stay unbiased; gives tests for choosing the signal. https://arxiv.org/abs/2509.26000
- **Le Critique** (Venkatraman, Dinot, Aitchison, et al.; v1 2026-08-17): privileged value functions that add token-level signal without biasing the objective, and TETHER, which interpolates between group and value baselines, for LLM RL. https://arxiv.org/abs/2608.16739
- **StateAlign** (Zhexi Lu et al.; v1 2026-11-12; HTML read). Constructs paired task states (Exec vs Need_User, Need_Fetch, Infeasible, Stop) from one source trajectory. Token contrast r = clip(log pi_0(y|state) - log pi_0(y|paired state), -1, 1) under the frozen base model; objective J = J_out + 0.1 J_pair. The policy never sees the paired context; there are no forks. BFCL v4 multi-turn 45.5% to 55.2% on Qwen3-8B. https://arxiv.org/abs/2611.16304

### 3.4 Branching at uncertain steps (text-prefix trees, no environment state restore)

TreeRL [LIT C8], Tree-GRPO [LIT H4], R3L [LIT H6, re-read: restart at the diagnosed pivot turn in text; tokens before the pivot masked; positive advantages scaled by 3], plus:
- **TreeRPO** (Zhicheng Yang et al. 2025): sibling nodes under one parent form a group; node reward = mean of child rewards; single-turn math. https://arxiv.org/abs/2506.05183
- **ARPO** (Guanting Dong et al. 2025; arXiv preprint): branches Z extra paths when token entropy rises after a tool response; shared-prefix tokens get averaged advantages (hard) or rely on identical ratios (soft). https://arxiv.org/abs/2507.19849
- **APPO** (Xucong Wang et al.; v2 2026-07-31; "work in progress"): branching at fine-grained decision points scored by uncertainty and likelihood gain, with procedure-level advantage scaling. https://arxiv.org/abs/2606.12384
- **TSR** (Djuhera et al.; ICLR 2026 Workshop on Lifelong Agents per arXiv): training-time tree search to build better trajectories. https://arxiv.org/abs/2602.11767
- **SPO** (Yiran Guo et al.; NeurIPS 2025 per arXiv): segment-level MC advantages (chain and tree variants). https://arxiv.org/abs/2505.23564
- **T-STAR** (Yu Li et al. 2026): merges similar steps into a tree, back-propagates rewards for step advantages, Bradley-Terry loss on critical steps. https://arxiv.org/abs/2604.07165
- **CRBC** (Yangyang Ren et al. 2026): Bellman fixed point over the merged group graph; step advantage Q - V plus trajectory advantage. https://arxiv.org/abs/2609.35082
- **TRACE** (Heming Zou et al.; EMNLP 2026 per arXiv): ReAct turns as tree nodes; a predictor of conditional success allocates rollouts to prefixes likely to give mixed outcomes. https://arxiv.org/abs/2606.11119
- **Gradient Coupling** (Jingyu Liu et al.; first versions titled "Rethinking Reward Miscalibration of GRPO in Agentic RL"): similar states cause destructive gradient interference; adds action classification to separate embeddings of good and bad actions. https://arxiv.org/abs/2509.23870

---

## 4. Candidate C: closest prior work (item 9)

**SAMPLR, Grounding Aleatoric Uncertainty for Unsupervised Environment Design** (Minqi Jiang, Michael Dennis, Jack Parker-Holder, Andrei Lupu, et al.; NeurIPS 2022 per arXiv; read on ar5iv).
- Problem: curriculum-induced covariate shift (CICS). When the curriculum shifts the distribution of aleatoric parameters that the trajectory does not reveal, the optimal policy under the training distribution differs from the one under the ground-truth distribution. Example: oversampling ice teaches a car to drive slowly in fair weather.
- Fix: in replayed levels, replace each transition with a fictitious one. Sample s' from the belief B(s | tau) = sum_theta' P(s | tau, theta') P(theta' | tau) under the ground-truth distribution, then use T(s', a) and R for the next state and reward. Built on PLR. Requires a resettable simulator and knowledge of the ground-truth distribution.
- Relation to ActionProof: the hidden commit outcome behind a byte-identical ack is exactly an aleatoric parameter in this sense. A recovery-start or frontier scheduler that oversamples faulted states is a CICS source. https://arxiv.org/abs/2207.05219

**Depth Before Breadth: Reverse-to-Forward Curriculum with Advantage Decomposition for LLM Agent** (Siyuan Du et al.; arXiv preprint, v1 2026-11-18; HTML read).
- Capability region C_eta(pi) = {contexts c : p_pi(c) >= eta} (eta 0.6 ALFWorld, 0.78 WebShop). Reverse stage: start from contexts along successful reference trajectories, moving back with stride d once success at the current start reaches eta. Forward stage: start from the original context with policy-generated prefixes, which can reach off-reference states; advance when endpoint diversity plateaus. Environments are reset to intermediate contexts.
- Paired advantages: A^L(e) = V^(e) - prefix-weighted mean V^ (credit for reaching a better state) and A^R(e, j) = R_ej - leave-one-out mean of suffixes from the same endpoint (credit for completing).
- No start-distribution correction (per fetched HTML). If all suffixes fail, both advantages vanish.
- Setup: ALFWorld, WebShop; Qwen2.5-1.5B and 7B; baselines PPO, RLOO, GRPO, GiGPO, CRPS. 90.1%/96.1% and 80.2%/82.6%.
- Relation to C: an adaptive success-threshold frontier of start states for LLM agents, published three days before this audit. https://arxiv.org/abs/2611.24697

**RAIL, Recoverability-aware Rollout Intervention Learning** (Zheyuan Zhang et al.; arXiv preprint, v1 2026-08-05; HTML read).
- An intervention picks a branch budget m in {4, 8, 12} at an anchor state and a decoding regime. "Recoverability" = expected gain in group reward variance from intervening. A contextual bandit on 26 features (anchor entropy, position, reward mean and std, distance to 0.5, failed-anchor count, tool diversity and history projections) is trained on (z, b, gain - lambda cost) traces, first from a heuristic "shadow" phase, then live with a utility gate and exploration.
- The GRPO update is unchanged. No distribution correction is discussed (per fetched HTML).
- Setup: AgentBench-OS/DB, WebShop, ToolQA-Coffee; Qwen3-4B; baselines GRPO, ARPO, AEPO, VIP, TAMPO, Tree-GRPO. https://arxiv.org/abs/2608.05080

**NoisyAgent, Learning to Act under Noise** (Yuxin Chen et al.; arXiv preprint, v1 2026-05-26; HTML read). Tool noise: failures, incomplete, misleading (silently wrong), redundant; plus user noise. Schedule: when the clean-minus-noisy success gap falls below theta = 0.05, raise the difficulty and share of that noise type, capped at 50% of rollouts. GRPO with advantages normalised separately within clean and noisy groups. Binary reward from a Claude-Sonnet-4.5 verifier that checks final state. Qwen3-8B and 32B; baselines GRPO, DAPO, GSPO. Relation: adaptive fault-exposure curriculum plus per-subgroup normalisation. https://arxiv.org/abs/2605.27209

**Bench2Robust** (Chaoran Chen et al.; arXiv preprint, v1 2026-08-12, v2 2026-10-11; HTML read). Nine failure modes: six explicit (timeout, rate limit, server, auth, malformed, schema drift) and three silent corruptions (partial data, stale values, factual errors). P(clean) = 0.60 per call. Scenarios S1 retry, S2 switch, S3 impossible. RL curriculum in five fixed phases, with advancement gated on diagnostic thresholds. DAPO with KL. Reward 1.0 eff rep if completed, else 0.3 (matched/required) rep; no positive term for correct abstention. States the simplifying assumption that retry after an explicit-signal failure does not risk a duplicate mutation. tau2-bench and BFCL multi-turn; Qwen3 4B to 235B and others. Also an inference-time Bayesian Tool Memory (Beta posteriors per tool and error type). https://arxiv.org/abs/2608.11977

**Online difficulty and learnability selection for LLM RL:**
- **LILO**, Learning to Reason at the Frontier of Learnability (Foster, Sims, Forkel, Fellows, Foerster; arXiv page lists no venue; v6 2026-04-30): prioritises questions with high success variance p(1-p); works with PPO and VinePPO. https://arxiv.org/abs/2502.12272
- **Online Difficulty Filtering for Reasoning Oriented RL** (Bae et al.; EACL 2026 long, pp. 700-719, verified at ACL Anthology 2026.eacl-long.30): expected policy improvement is lower-bounded by the variance of task success probabilities; balanced filtering. https://aclanthology.org/2026.eacl-long.30
- **MoPPS**, Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Models? (Yun Qu et al.; arXiv preprint, v5 2026-01-10): per-prompt Beta posterior over success rate with Thompson sampling, which covers both failure probability and uncertainty. https://arxiv.org/abs/2507.04632
- **PAC**, Progress-Augmented Advantage Curriculum (Yuanqiang Yu et al.; EMNLP 2026, pp. 25825-25843, verified at ACL Anthology 2026.emnlp-main.1127): advantage-derived learnability plus recent reward gain, with Thompson sampling across tasks. https://aclanthology.org/2026.emnlp-main.1127/
- **Self-Evolving Curriculum** (Xiaoyin Chen et al. 2025): non-stationary bandit over categories with absolute advantage as reward. https://arxiv.org/abs/2505.14970
- **Actor-Curator** (Zhengyao Gu et al. 2026): neural curator trained as a policy-improvement bandit with regret guarantees. https://arxiv.org/abs/2602.20532
- **Prioritized Replay for RL Post-training** (Mehdi Fatemi 2026): priority from success statistics favouring intermediate difficulty, heap sampling and periodic retesting. https://arxiv.org/abs/2601.02648
- **VCRL** (Guochao Jiang et al. 2025): group-reward variance as difficulty signal plus replay memory. https://arxiv.org/abs/2509.19803
- DAPO dynamic sampling [LIT A2], GRESO [LIT H2], Selective Rollout [LIT H3].

**Classic curricula and UED:**
- **Prioritized Level Replay** (Jiang, Grefenstette, Rocktaschel; arXiv page lists no venue): replay levels by TD-error-based learning potential plus staleness. https://arxiv.org/abs/2010.03934
- **PAIRED** (Dennis et al.; arXiv page lists no venue): adversary maximises protagonist-antagonist regret. https://arxiv.org/abs/2012.02096
- **ACCEL** (Parker-Holder et al.; arXiv page lists no venue): regret-based replay plus level editing at the capability frontier. https://arxiv.org/abs/2203.01302
- **Teacher-Student Curriculum Learning** (Matiisen, Oliver, Cohen, Schulman 2017): pick tasks with the steepest learning curve slope, revisit forgotten ones. https://arxiv.org/abs/1707.00183
- **ALP-GMM** (Portelas et al.; CoRL 2019 per arXiv): absolute learning progress modelled with GMMs over environment parameters. https://arxiv.org/abs/1910.07224
- **Automated Curriculum Learning for Neural Networks** (Graves et al. 2017): learning-progress signals fed to a bandit. https://arxiv.org/abs/1704.03003
- **CeSoR**, Efficient Risk-Averse RL (Greenberg, Chow, Ghavamzadeh, Mannor; NeurIPS 2022 per arXiv): cross-entropy sampling that oversamples high-risk environment conditions, plus a soft-risk schedule. This is the severity-weighted sampling component. https://arxiv.org/abs/2205.05138
- Reverse Curriculum [LIT C3], Go-Explore [LIT C4], R^3 [LIT C5], failure-prefix conditioning [LIT C7], RAGEN diverse initial states [LIT A7].

**Failure-driven task generation for agents:**
- **SENTINEL** (Ziyi Wang et al.; arXiv preprint, v1 2026-06-11; HTML read): an LLM controller clusters failure patterns, a proposer generates executable tasks that exercise them, prioritised by failure-pattern frequency; GRPO; DBCheck state reward plus penalties, including a repeated-identical-call penalty; faults are visible. Does not restart from failure states. tau2-bench Retail with Qwen3-4B-Thinking: Pass^1 66.4 to 74.9. https://arxiv.org/abs/2606.12908
- **WebRL** (Qi et al.; ICLR 2025 per arXiv): new tasks generated from failed attempts. https://arxiv.org/abs/2411.02337
- Fission-GRPO's LIFO buffer of recent failures (section 2.1).

---

## 5. Candidate B: related constrained and multi-reward GRPO

- **GDPO** (Shih-Yang Liu et al.; NVIDIA tech report, v1 2026-01-08): normalising summed rewards collapses distinct reward combinations into identical advantages; normalise each reward in the group separately. Includes tool calling. https://arxiv.org/abs/2601.05242
- **MO-GRPO** (Ichihara et al.; TACL vol. 14, pp. 1867-1882, verified at ACL Anthology 2026.tacl-1.84): GRPO reward-hacks one objective in multi-objective problems; reweight rewards by their variances. https://aclanthology.org/2026.tacl-1.84/
- **CorrGRPO** (Wenbin Hu et al.; v1 2026-09-29): normalise pairwise covariances into correlations so large correlated rewards do not dominate; tool calling and agent security tasks. https://arxiv.org/abs/2609.36820
- **SafeDojo** (Kai Tang et al.; CoRL 2026 per arXiv): Lagrangian-constrained GRPO with separately normalised task reward and safety cost, on a video world model for VLA policies. https://arxiv.org/abs/2606.20698
- **Web Agent Agentic RL Decision Model Under Multi-Cost and Failure Risk** (Qianli Ma et al.; BDICN 2026 per arXiv): Lagrangian budgets over multiple costs plus CVaR on failure for web agents. https://arxiv.org/abs/2610.23857
- **Abstention as an Action Can Kill Both the Reward Gradient and the KL Anchor** (Xujun Che et al.; v1 2026-07-31): with abstention as an action, the reward gradient and KL restoring force share a saturating factor and die together; group normalisation moves the learned threshold to 0.5. https://arxiv.org/abs/2608.00301
- **ToolRLA** (Pengbo Liu et al. 2026): multiplicative reward over format, tool selection, parameters and compliance; +7 points over additive. https://arxiv.org/abs/2603.01620
- **EAPO**, Learning When Not to Act: Mitigating Tool Abuse in Agentic RL (Liuji Chen et al.; under review): tool-free trajectories, difficulty-based shaping against redundant tool calls, confidence-aware token reweighting. https://arxiv.org/abs/2606.02132
- **Multi-Turn RL for Tool-Calling Agents with Iterative Reward Calibration** (Modecrua et al. 2026): naive dense per-turn rewards cost up to 14 points; calibrated rewards give read-only calls zero and penalise non-golden state-changing calls. https://arxiv.org/abs/2604.02869
- **FSPO** (Miaobo Hu et al. 2026): risk-to-go and multi-resource budget feasibility for adaptive RL post-training controllers (marginal). https://arxiv.org/abs/2610.02828
- Also: CPO [LIT D1], RCPO [LIT D2], Safe RLHF [LIT D3], CVaR PG [LIT D4], lexicographic RL [LIT D6], MOSAIC [LIT D8], TruthRL [LIT E2], OTC [LIT E3], SURE-Act (section 3.1, utility s - lambda c - eta r with an irreversible-error term).

---

## 6. Side effects, idempotency, duplicates, false success and stateful-tool RL, 2025-2026 (item 10)

Training papers with tool faults:
- NoisyAgent and Bench2Robust (section 4) train with **silent output corruption** (misleading, partial, stale or wrong values). Neither models a silent write commit, duplicate delivery or a byte-identical ack. Bench2Robust explicitly assumes that a retry after an explicit failure cannot duplicate a mutation.
- SENTINEL (state reward, visible faults), Fission-GRPO and ReGRPO (explicit errors), ReflectPO (stateful BFCL with forks), StateAlign (recovery after tool errors as a side metric), AgentRobustBench (explicit transient errors).
- **GracefulRL**, Training Language Models to Fail Gracefully (Arora et al.; v1 2026-10-14): outcome-level RL that keeps success where possible and executes a safe fallback when acting is unreliable, with no fallback annotations. https://arxiv.org/abs/2610.20401
- **EigenData** (Gao et al.; "submitted to ICML 2026" per arXiv): self-evolving synthetic tool dialogues with executable verifiers, GRPO with dynamic filtering, tau2-bench. https://arxiv.org/abs/2601.22607
- **COVERT** (Siyuan Xu et al. 2026): oracle-preserving synthetic tool environments with distractors and unreliable tool outputs for RL. https://arxiv.org/abs/2604.09813
- **PALADIN** (Vuddanti et al. 2025): SFT on 50K+ failure-injected recovery trajectories, plus retrieval of failure exemplars at inference. https://arxiv.org/abs/2509.25238

Evaluation and runtime papers (no policy training):
- LIMBO [LIT F3, re-fetched; evaluation only, confirmed]: 0.5% duplicates when the ack is lost but verification is possible; 56-74% when in flight or redelivered; idempotency keys cut duplicates from 28% to 4%.
- **RetryBench** (You Liu et al.; v4 2026-11-04): retry decisions under timeouts depend strongly on the wording of the idempotency key. https://arxiv.org/abs/2610.15884
- **Evidence-Guided Recovery for AI Agents with Ambiguous Tool Outcomes and Partial Execution** (Genliang Zhu et al.; v1 2026-10-30): planning over states consistent with evidence plus a separate effect adjudicator and terminal certificate. https://arxiv.org/abs/2610.41453
- **Outcome Monitors** (Panthi, Abdelfattah, et al.; v1 2026-08-19): runtime detection of outcome-contract violations that masquerade as valid outputs, with recovery receipts. https://arxiv.org/abs/2608.19303
- **CASEL** (Yanyan Zhang et al.; CBASE 2026 per arXiv): runtime choice among execute, verify, sandbox, clarify, approve, abstain from failure probability, consequence, reversibility, ambiguity and evidence freshness; simulated. https://arxiv.org/abs/2610.36570
- **Tool-Calling LLM Agents Under Realistic MCP Faults** (Iannillo et al.; NeurIPS 2026 poster per arXiv): scale does not buy fault tolerance. https://arxiv.org/abs/2610.32555
- **Don't Blindly Trust It** (Chubin Zhang et al. 2026): misleading tool feedback can be worse than none ("value inversion"). https://arxiv.org/abs/2606.21409
- **Recovery Is a Decision, But Whose?** (Sathvik Subramanian; NeurIPS 2026 Workshop on Evaluation of Interactive Agents, verified at neurips.cc event 166916; no arXiv link on the page): replay-based evaluation of tool-failure triage is reversed by live agents. This is a direct caution for recovery starts built from replayed prefixes. https://neurips.cc/virtual/2026/166916
- **Same Failure, Different Cause** (Abhimanyu Prasad; NeurIPS 2026 workshop on grounded user simulation, verified at neurips.cc poster 169228): simulator design hides or exposes ungrounded tool calls. https://neurips.cc/virtual/2026/poster/169228
- Confident Closing [LIT F4], Verified Tool Calls [LIT H17], FTA [LIT H18], tau-bench [LIT F1].

Reversibility:
- **REVERSAL-BENCH** (Shaik, Venkataraman 2026): reset-free RL agents get absorbed into irrecoverable states as irreversibility rises (robotics). https://arxiv.org/abs/2609.17745
- **Learning to Undo** (Sorstkins et al. 2025; submitted to PLOS ONE): a reversibility measure Phi and selective rollback in tabular RL. https://arxiv.org/abs/2510.14503
- Stepwise relative reachability [LIT H9], attainable utility preservation [LIT H10].

Hidden state and learning:
- **When Does Reward Teach State?** (Allchin et al. 2026): high reward without learning the hidden automaton state "occurs but is rare"; reward alone cannot confirm state learning. https://arxiv.org/abs/2607.11953
- **Where Success Breaks** (Yanzhe Chen et al. 2026; marginal): discovering failure boundaries with simulator states for VLA policies. https://arxiv.org/abs/2609.06114

---

## 7. (a) Comparison table

Columns: problem; update rule; what the policy (P) and evaluator (E) observe; handling of recovery states or irreversibility; closest candidate; redundancy risk for that candidate (High: the mechanism is already implemented; Med: a main component is implemented; Low: context only).

| Paper | Problem | Update rule | Observes | Recovery / irreversibility | Closest | Risk | URL |
|---|---|---|---|---|---|---|---|
| Fission-GRPO (ACL 2026) | recovery after explicit tool errors | failed rollout + simulator feedback becomes new prompt; G' rollouts; same GRPO, in-group z | P: explicit error text; E: reference call | text restart from failure; no fork; side effects ignored | Stage 3, C | High for Stage 3 mechanism; Low for A, B | https://aclanthology.org/2026.acl-long.1880/ |
| ReGRPO | VLM tool fragility | SFT on RoT triplets; GRPO K=5, success - eta x reflection length | P: explicit failures; E: env success | near-miss failures executed offline; no restarts | Stage 3 | Low | https://arxiv.org/abs/2606.31392 |
| Constrained GRPO [LIT D7] | constraints in GRPO | per-component z, then lambda-weighted sum; dual ascent | E: indicator costs | none | B | High | https://arxiv.org/abs/2602.05863 |
| Safety as a Budget | harm and over-refusal budgets | per-objective z or unscaled; isotonic-calibrated control channel | E: judge on text | none; single turn | B | High | https://arxiv.org/abs/2611.16628 |
| Proof-of-Use (EMNLP 2026) | decorative tool calls | GRPO; citation, perturbation, support, correctness rewards | P: tool outputs; E: citation and perturbation tests, LLM judge | read-only tools | none (evidence use) | Low | https://aclanthology.org/2026.emnlp-main.1664/ |
| Reward Hacking Benchmark (ICML 2026) | exploit propensity | evaluation only | E: hidden recomputation | false completion markers, skipped verification | B (false completion) | Low | https://proceedings.mlr.press/v306/thaman26a.html |
| AgentRobustBench / When Simulation Lies (NeurIPS 2026 E&D) | sim-to-real gap | ToolRL GRPO on uniformly perturbed data | P: explicit errors; E: tool/param match | one turn after explicit error; silent no-op deferred | C | Low | https://arxiv.org/abs/2605.11928 |
| TRPO vine (ICML 2015) | variance reduction | reset to visited states, K actions, CRN, Q per action | full state | needs arbitrary resets | A | High | https://arxiv.org/abs/1502.05477 |
| CRR (NeurIPS 2026 per arXiv) | SWE step credit | snapshot at entropy x action-type steps; R(tau) - R(tau~) = Q - E_q Q; GRPO elsewhere | P: tool outputs; E: hidden tests | determinism filter; no irreversibility | A | High | https://arxiv.org/abs/2609.33875 |
| BPO (WAIC 2026 per arXiv) | sandbox agent RL | snapshot at top-entropy steps; K siblings; LOO sibling baseline | full state restore | no safety handling | A | High | https://arxiv.org/abs/2607.14171 |
| ReflectPO | error accumulation | snapshots at recovery anchors; parent-child return diff; recovery gain | P: hints at generation only | forks in stateful BFCL; vs Fission-GRPO | A, Stage 3 | High | https://arxiv.org/abs/2610.30933 |
| BiPACE | step credit grouping | same-action-key mean minus cluster mean, + episode advantage | groups by actor hidden states | no forks | A | High | https://arxiv.org/abs/2606.25556 |
| RTMC | SWE step credit | MC Q(s,a) - V(s) over tool-category action classes | state-action signatures | no forks | A | High | https://arxiv.org/abs/2604.11037 |
| GiGPO [LIT A9] | step credit | anchor-state groups; A_E + omega A_S | exact state hash | repeated states only | A | High | https://arxiv.org/abs/2505.10978 |
| COMA [LIT C2] | multi-agent credit | Q(s,u) - sum pi(u'|tau) Q(s,u') | E: global state; P: history | n/a | A | High (baseline form) | https://arxiv.org/abs/1705.08926 |
| VinePPO [LIT C6] | value estimation | MC values from resumed prefixes | full prefix | n/a | A | Med | https://arxiv.org/abs/2410.01679 |
| ELPO | first irrecoverable step | binary-search prefix rollouts; branch z + trajectory z; relaxed clip | replayed tool outputs | targets irrecoverable step | A | Med-High | https://arxiv.org/abs/2602.09598 |
| SURE-Act (Agents' Last Doubt) | act vs ask vs escalate | clone env, force each meta-action, s - lambda c - eta r; ranking controller | E: critical-error labels | explicit Commit-Irreversible class | A, B | High for forked action-class scoring; Med as RL | https://arxiv.org/abs/2610.24194 |
| TriGRPO | rare successes | hindsight branch with banked action; R(tau_j) - R(tau_i) | text replay | says irreversible domains hard | A | Med | https://arxiv.org/abs/2611.02706 |
| CSO (ACL 2026 Findings per arXiv) | critical steps | verified alternative actions, preference learning | E: PRM, execution | starts from failed trajectories | A | Med | https://arxiv.org/abs/2602.03412 |
| MAGMA-GEN (CoRL 2026 per arXiv) | ambiguous failures | keep coach fixes only if re-execution from same state improves | E: privileged coach | identical-state re-execution | A | Low-Med | https://arxiv.org/abs/2609.20056 |
| StateAlign | state-inconsistent actions | base-model contrast between paired states + outcome GRPO | P: one state only | recovery after errors improves | A (contrast) | Low-Med | https://arxiv.org/abs/2611.16304 |
| Wu et al. 2018 | PG variance | action-dependent factorised baseline | MDP | n/a | A | Med | https://arxiv.org/abs/1803.07246 |
| Tucker et al. 2018 | baseline audit | shows learned action-dependent baselines add little | MDP | n/a | A (warning) | Med | https://arxiv.org/abs/1802.10031 |
| Q-Prop (ICLR 2017 per arXiv) | sample efficiency | critic Taylor expansion as control variate | MDP | n/a | A | Low | https://arxiv.org/abs/1611.02247 |
| Counterfactual Credit Assignment | luck vs skill | future-conditional baselines | hindsight features | n/a | A | Low | https://arxiv.org/abs/2011.09464 |
| RUDDER | delayed reward | return decomposition, redistribution | MDP | n/a | A | Low | https://arxiv.org/abs/1806.07857 |
| Baisero and Amato (AAMAS 2022 per arXiv) | asymmetric critics | history-state critic is unbiased; state critic is not | E: state; P: history | n/a | A (bias) | Med | https://arxiv.org/abs/2105.11674 |
| Informed Asymmetric AC (ICML 2026 per arXiv) | privileged signals | unbiased critics on partial privileged signals | E: privileged; P: obs | n/a | A (bias) | Low-Med | https://arxiv.org/abs/2509.26000 |
| Le Critique | LLM RL variance | privileged value functions; TETHER interpolation | E: privileged | n/a | A | Low-Med | https://arxiv.org/abs/2608.16739 |
| TreeRL [LIT C8] | tree rollouts | branch at uncertain tokens | text | n/a | A | Med | https://arxiv.org/abs/2506.11902 |
| Tree-GRPO [LIT H4] | agent tree rollouts | shared prefixes, step signal | text | n/a | A | Med | https://arxiv.org/abs/2509.21240 |
| R3L [LIT H6] | failure restart | restart at pivot turn, mask prefix | text | n/a | A, Stage 3 | Med | https://arxiv.org/abs/2601.03715 |
| TreeRPO | math step rewards | sibling groups, node reward = child mean | text | n/a | A | Low-Med | https://arxiv.org/abs/2506.05183 |
| ARPO | post-tool uncertainty | entropy-triggered branching; shared-token advantage | text | n/a | A | Med | https://arxiv.org/abs/2507.19849 |
| APPO | fine-grained credit | branch score + procedure-level scaling | text | n/a | A | Med | https://arxiv.org/abs/2606.12384 |
| TSR | rollout quality | training-time tree search | text | n/a | A | Low | https://arxiv.org/abs/2602.11767 |
| SPO (NeurIPS 2025 per arXiv) | segment credit | segment MC advantages | text | n/a | A | Low-Med | https://arxiv.org/abs/2505.23564 |
| T-STAR | step credit | cognitive tree, BT loss on critical steps | text | n/a | A | Low-Med | https://arxiv.org/abs/2604.07165 |
| CRBC | step credit | Bellman closure over merged group | text | n/a | A | Med | https://arxiv.org/abs/2609.35082 |
| TRACE (EMNLP 2026 per arXiv) | rollout budget | predicted success guides prefix allocation | text | n/a | A, C | Med | https://arxiv.org/abs/2606.11119 |
| BunshinRL | fork cost | kernel-level workspace forking | n/a | n/a | A (infra) | Low | https://arxiv.org/abs/2611.10453 |
| Gradient Coupling | similar-state interference | action classification head | n/a | n/a | A | Low | https://arxiv.org/abs/2509.23870 |
| SAMPLR (NeurIPS 2022 per arXiv) | curriculum-induced covariate shift | fictitious transitions from ground-truth posterior; PLR | P: partial; E: aleatoric params | resettable simulator | C (and A) | High for the correction concept | https://arxiv.org/abs/2207.05219 |
| Depth Before Breadth | long-horizon agents | reverse-to-forward start frontier, thresholds; paired L/R advantages | resets to intermediate contexts | no distribution correction | C | High | https://arxiv.org/abs/2611.24697 |
| RAIL | rollout allocation | bandit picks branch budget at anchors by predicted variance gain | rollout stats | no correction | C, A | High | https://arxiv.org/abs/2608.05080 |
| NoisyAgent | noisy tools | progressive noise when gap < theta; GRPO normalised per clean/noisy group | E: LLM verifier on final state | silent misleading outputs | C, B | Med-High | https://arxiv.org/abs/2605.27209 |
| Bench2Robust | tool failure recovery | phase-gated retry/switch/impossible curriculum; DAPO | E: task evaluator | silent corruptions; assumes no duplicate on retry | C | Med | https://arxiv.org/abs/2608.11977 |
| PLR | level selection | TD-error score + staleness replay | n/a | n/a | C | High | https://arxiv.org/abs/2010.03934 |
| PAIRED | UED | regret-maximising adversary | n/a | n/a | C | Med | https://arxiv.org/abs/2012.02096 |
| ACCEL | UED | regret replay + level edits at frontier | n/a | n/a | C | Med | https://arxiv.org/abs/2203.01302 |
| Teacher-Student CL | task selection | learning-curve slope bandit | n/a | n/a | C | High (LP) | https://arxiv.org/abs/1707.00183 |
| ALP-GMM (CoRL 2019 per arXiv) | task selection | absolute learning progress | n/a | n/a | C | High (LP) | https://arxiv.org/abs/1910.07224 |
| Graves et al. 2017 | syllabus | learning-progress bandit | n/a | n/a | C | Med | https://arxiv.org/abs/1704.03003 |
| CeSoR (NeurIPS 2022 per arXiv) | risk-averse RL | cross-entropy oversampling of high-risk conditions | env parameters | n/a | C (severity) | Med-High | https://arxiv.org/abs/2205.05138 |
| LILO | zero-variance prompts | sample by p(1-p) | n/a | n/a | C | High | https://arxiv.org/abs/2502.12272 |
| Online Difficulty Filtering (EACL 2026) | prompt selection | balanced intermediate-difficulty filter | n/a | n/a | C | High | https://aclanthology.org/2026.eacl-long.30 |
| MoPPS | prompt selection | Beta posterior + Thompson sampling | n/a | n/a | C (uncertainty) | High | https://arxiv.org/abs/2507.04632 |
| PAC (EMNLP 2026) | multi-task schedule | learnability + reward gain, Thompson | n/a | n/a | C (LP) | High | https://aclanthology.org/2026.emnlp-main.1127/ |
| SEC | curriculum | bandit on absolute advantage | n/a | n/a | C | Med | https://arxiv.org/abs/2505.14970 |
| Actor-Curator | curriculum | policy-improvement bandit | n/a | n/a | C | Med | https://arxiv.org/abs/2602.20532 |
| Prioritized Replay for RL Post-training | prompt priority | success-statistics priority, retesting | n/a | n/a | C | Med | https://arxiv.org/abs/2601.02648 |
| VCRL | difficulty control | reward-variance sampling + replay | n/a | n/a | C | Med | https://arxiv.org/abs/2509.19803 |
| SENTINEL | task distribution | failure-pattern-driven task generation; GRPO | P: visible faults; E: DB state | new tasks, not restarts | C | Med | https://arxiv.org/abs/2606.12908 |
| WebRL (ICLR 2025 per arXiv) | web agent tasks | tasks from failures + ORM | E: ORM | n/a | C | Low-Med | https://arxiv.org/abs/2411.02337 |
| Reverse Curriculum [LIT C3] | sparse reward | starts with intermediate success | state resets | n/a | C | High | https://arxiv.org/abs/1707.05300 |
| Go-Explore [LIT C4] | exploration | archive + return to states | state restore | n/a | C | Med | https://arxiv.org/abs/1901.10995 |
| R^3 [LIT C5] | outcome-only RL | start late in demo, slide back | text prefix | n/a | C | Med | https://arxiv.org/abs/2402.05808 |
| Failure-prefix conditioning [LIT C7] | saturated problems | condition on failure prefixes | text prefix | n/a | C, Stage 3 | High for Stage 3 | https://arxiv.org/abs/2601.20829 |
| DAPO [LIT A2] | zero-variance groups | dynamic sampling | n/a | n/a | C | Med | https://arxiv.org/abs/2503.14476 |
| GRESO [LIT H2] | rollout cost | skip predicted zero-variance prompts | n/a | n/a | C | Med | https://arxiv.org/abs/2506.02177 |
| GDPO | multi-reward GRPO | per-reward group normalisation | n/a | n/a | B | High | https://arxiv.org/abs/2601.05242 |
| MO-GRPO (TACL 2026) | multi-objective hacking | variance-based reward reweighting | n/a | n/a | B | High | https://aclanthology.org/2026.tacl-1.84/ |
| CorrGRPO | multi-reward scale | correlation-normalised GRPO | n/a | n/a | B | Med | https://arxiv.org/abs/2609.36820 |
| SafeDojo (CoRL 2026 per arXiv) | VLA safety | Lagrangian GRPO, separately normalised cost | E: learned safety head | n/a | B | High | https://arxiv.org/abs/2606.20698 |
| Web agent multi-cost (BDICN 2026) | budgets, failure risk | Lagrangian + CVaR | E: costs | failure-risk penalty | B | Med | https://arxiv.org/abs/2610.23857 |
| Abstention collapse (2608.00301) | error-penalised RL | diagnosis + proper-scoring confidence fix | E: correctness | abstention collapse | B | Med | https://arxiv.org/abs/2608.00301 |
| ToolRLA | tool reward design | multiplicative reward decomposition | E: rubric | compliance term | B | Low-Med | https://arxiv.org/abs/2603.01620 |
| EAPO | tool overuse | tool-free trajectories, redundancy shaping | E: correctness | unnecessary calls | B | Med | https://arxiv.org/abs/2606.02132 |
| MT-GRPO reward calibration | per-turn rewards | calibrated turn rewards | E: golden actions | penalise non-golden writes | B | Med | https://arxiv.org/abs/2604.02869 |
| FSPO | budgeted RL control | risk-to-go, feasibility certificate | n/a | n/a | B | Low | https://arxiv.org/abs/2610.02828 |
| CPO / RCPO / Safe RLHF [LIT D1-D3] | constrained RL | trust region / Lagrangian | E: cost | n/a | B | High (concept) | https://arxiv.org/abs/1805.11074 |
| MOSAIC [LIT D8] | irreversible harmful tools | plan-check-act-or-refuse, preference RL | E: judge | refusal as action | B | Med | https://arxiv.org/abs/2603.03205 |
| TruthRL [LIT E2] | abstention | ternary reward GRPO | E: correctness | abstention at 0 | B | Med | https://arxiv.org/abs/2509.25760 |
| GracefulRL | safe fallback | outcome-level RL for fallback | E: outcome | fallback emerges | B | Low-Med | https://arxiv.org/abs/2610.20401 |
| LIMBO [LIT F3] | exactly-once | evaluation only | E: services state | duplicates, late commits | setting | Low | https://arxiv.org/abs/2609.29095 |
| RetryBench | retry permission | evaluation only | prompt variants | idempotency keys | setting | Low | https://arxiv.org/abs/2610.15884 |
| Evidence-Guided Recovery | ambiguous outcomes | planner + effect adjudicator | E: certificate | partial execution | setting | Low | https://arxiv.org/abs/2610.41453 |
| Outcome Monitors | silent tool failures | runtime contract monitor | runtime | recovery receipts | setting | Low | https://arxiv.org/abs/2608.19303 |
| CASEL | consequence-aware execution | runtime mode selection | simulated | reversibility factor | B | Low | https://arxiv.org/abs/2610.36570 |
| MCP faults (NeurIPS 2026 per arXiv) | fault tolerance | evaluation only | n/a | permanent faults | setting | Low | https://arxiv.org/abs/2610.32555 |
| Don't Blindly Trust It | unreliable feedback | evaluation only | n/a | n/a | setting | Low | https://arxiv.org/abs/2606.21409 |
| Recovery Is a Decision (NeurIPS 2026 WS) | triage evaluation | live vs replay evaluation | n/a | replay results reversed live | Stage 3, C | Med (methodological) | https://neurips.cc/virtual/2026/166916 |
| Same Failure, Different Cause (NeurIPS 2026 WS) | simulator provenance | evaluation only | n/a | n/a | setting | Low | https://neurips.cc/virtual/2026/poster/169228 |
| REVERSAL-BENCH | reset-free RL | benchmark | reset oracle | irreversibility cliff | setting | Low | https://arxiv.org/abs/2609.17745 |
| Learning to Undo | reversibility | penalty + selective rollback | tabular | rollback | B | Low | https://arxiv.org/abs/2510.14503 |
| EigenData | tool-agent data | synthetic data + GRPO, dynamic filtering | E: executable verifiers | n/a | C | Low | https://arxiv.org/abs/2601.22607 |
| COVERT | RL environments | oracle-preserving augmentation | E: oracle | unreliable outputs | C | Low | https://arxiv.org/abs/2604.09813 |
| PALADIN | tool failure recovery | SFT on injected failures | P: explicit errors | n/a | Stage 3 | Low | https://arxiv.org/abs/2509.25238 |
| Hidden-automaton study | does reward teach state | probes on hidden automata | E: hidden state | n/a | setting | Low | https://arxiv.org/abs/2607.11953 |
| Where Success Breaks | VLA failure boundaries | simulator-state progress signals | E: sim state | n/a | C | Low | https://arxiv.org/abs/2609.06114 |

---

## 8. (b) Verdicts

### Candidate A: counterfactual state-forked branches

**Already done.**
1. *Fork the exact state and sample K actions to estimate Q(s,a).* TRPO vine (2015, with arbitrary resets and common random numbers). VinePPO (MC values from resumed prefixes). BPO (sandbox snapshots at high-entropy steps, K siblings, leave-one-out sibling baseline). CRR (container snapshots; an expectation result of the form Q(s,a) - E_q[Q(s,a~)]; hybrid with GRPO elsewhere). ReflectPO (snapshots at recovery anchors in stateful BFCL APIs, already compared with Fission-GRPO and GiGPO). ELPO (prefix rollouts to find the first irrecoverable step). SURE-Act (clones the environment, forces each meta-action including Commit-Irreversible, scores success minus cost minus critical error).
2. *Action-conditioned same-state baseline over action classes.* BiPACE's PACE Q-style form is Q^(s,a) over same-action-key peers minus the cluster mean. RTMC computes Q^(s,a) - V^(s) over tool-category action classes. COMA uses the pi-weighted counterfactual baseline. Wu et al. and Q-Prop are the classic action-dependent baselines. Tucker et al. report that learned action-dependent baselines gave no variance gain in practice.
3. *Step advantage on the decisive tokens plus an episode-level advantage.* GiGPO (A_E + omega A_S), BiPACE (A_ep + w A^S), CRR (hybrid), ELPO (lambda_tree mix), TriGRPO (root plus node groups).
4. *Choosing where to fork.* Entropy (BPO, ARPO, TreeRL, CRR). Action type with writes weighted highest (CRR gives edits weight 1.0). First error (ELPO, R3L). Hindsight success (TriGRPO). Recovery anchors (ReflectPO). A learned controller (RAIL).

**Algebraic reduction.** Let q be the empirical class frequencies among the K forked branches, and Q^(s,a) the mean return of branches in class a. Then sum_a' q(a'|s) Q^(s,a') is the mean return of all K branches, so A(s,a) = (class mean) - (sibling mean). This is BiPACE's Q-style estimator computed on forked siblings instead of clustered rollouts. Averaged over a class, it equals (K-1)/K times the class average of BPO's leave-one-out advantage; class pooling only replaces each branch's own return with its class mean, which averages out continuation noise within the class. With q = pi_theta(class | history) it is COMA's counterfactual baseline with Monte Carlo Q in place of a learned critic, and it needs forced branches for classes that were not sampled; CRR (tempered proposal excluding the factual action) and SURE-Act ("force then roll") already do this. Small point: the plain "class mean minus all-sibling mean" includes the branch's own return in the baseline; the leave-one-out forms (BPO, BiPACE diff-peer) avoid that 1/K bias.

**What could be new in ActionProof's setting.**
- *Forks under OEHF.* The forked Q is conditioned on the hidden state s; the policy conditions on history h. Baisero and Amato show that state-only critics are biased for history-conditioned policies, while history-state critics are not. If forks are triggered by a hidden-state label (the verifier sees a phantom ack), the extra gradient mass lands only on faulted hidden states. That over-weights the actions that are best when a fault is present (verify, escalate) relative to p(fault | h), which pushes toward over-verification at the deployment fault rate. If forks are triggered by an observation-defined event (every write ack, whatever the hidden outcome), the forked returns average over the true conditional distribution and estimate Q(h,a) without this bias. A "twin fork" variant re-samples the hidden commit outcome at the same history; that is SAMPLR's fictitious-transition idea applied to credit assignment. The extracted method sections of CRR, BPO, BiPACE, RTMC and ReflectPO do not discuss aliasing of this kind: CRR assumes deterministic forks, BPO assumes restore preserves the transition distribution, and BiPACE gives only a bisimulation bias bound. Confirm by reading their limitations sections before claiming this gap in a paper.
- *Irreversible and duplicate effects inside branches.* Forks make a duplicate or irreversible write safe to explore during training. SURE-Act already labels Commit-Irreversible branches, but uses them to train a controller, not as a policy-gradient advantage.

**A is redundant if:**
- it is presented as "fork the environment at critical steps, sample K actions, apply an action-conditioned same-state baseline to the decisive tokens plus the GRPO episode advantage" and evaluated by success gain over GRPO. That is CRR + BPO + BiPACE/RTMC + GiGPO;
- "safety-critical state" means a hidden fault label and no correction is applied. That is a CRR-style selector with a known asymmetric-information bias, not a contribution;
- the action classes are tool types or verify/retry/stop categories with no argument for why class pooling matters (RTMC classifies by tool category; BiPACE keys on the first tokens);
- the claimed benefit is variance reduction alone (Wu et al., Tucker et al. and BPO already quantify this).

### Candidate B: recovery-aware constrained GRPO

**Already done.**
- *Separate costs, per-component group normalisation, Lagrangian combination.* Constrained GRPO (indicator costs, per-component z-scores, lambda-weighted sum, proof that scalarise-then-standardise couples the multipliers). Safety as a Budget (standard GRPO normalisation cancels the multiplier; per-objective z or unscaled advantages; calibrated control channel; two-sided budgets against harmful action and over-refusal). SafeDojo (separately normalised reward and cost in Lagrangian GRPO). MO-GRPO, GDPO and CorrGRPO (per-reward normalisation variants; GDPO and CorrGRPO include tool calling). NoisyAgent (normalisation within clean and noisy subgroups).
- *The same cost types in agents.* Irreversible or critical actions: SURE-Act's eta r term, MOSAIC [LIT D8], the web-agent multi-cost Lagrangian with CVaR. False completion: the Reward Hacking Benchmark's sequence manipulation, Confident Closing [LIT F4], FTA [LIT H18], TruthRL's negative reward for wrong answers [LIT E2]. Unnecessary actions: OTC [LIT E3], EAPO, SENTINEL's repeated-call penalty, the calibrated penalty on non-golden state-changing calls (2604.02869).
- *Penalty failure modes under group normalisation.* 2608.00301 (abstention collapse; threshold moved to 0.5); Safety as a Budget (level vs ranking); ActionProof's own A6 sign flip (novelty scouts, G1).

**What could be new.**
- In ActionProof the false-completion cost depends on a hidden variable the policy cannot see. Under per-component z-scoring with G = 8, two observation-identical "stop and report done" rollouts can receive large advantages of opposite sign, depending only on the hidden commit draw. This is real, but it is a property of the OEHF setting and appears under any reward design, not just B.
- Dual variables updated on costs measured under the recovery-start distribution enforce the budget on that distribution, not on the deployment distribution (CICS applied to constraints). The fetched constrained-GRPO papers do not discuss this. It is really a C issue, discussed next.

**B is redundant if:**
- it is Constrained GRPO (or Safety as a Budget's two-channel version) with three indicator costs. That is K = 3 in Constrained GRPO's scalarised-advantage form, and calling it "recovery-aware" does not change the update;
- its empirical claim is that separate costs prevent collapse to always-refuse or always-verify (Safety as a Budget, Safe RLHF [LIT D3], RCPO [LIT D2], TruthRL [LIT E2]);
- its claim is that per-component normalisation beats scalar penalties (Constrained GRPO Theorem 4.1, GDPO, MO-GRPO).

### Candidate C: adaptive recovery-frontier scheduling

**Already done, signal by signal.**
- *Failure probability or intermediate success:* Reverse Curriculum [LIT C3], LILO, Online Difficulty Filtering, Prioritized Replay for RL post-training, VCRL, DAPO [LIT A2], GRESO [LIT H2].
- *Severity:* CeSoR (oversamples high-risk environment conditions), CVaR PG [LIT D4], the web-agent CVaR cost.
- *Uncertainty:* MoPPS and PAC (Beta posteriors with Thompson sampling).
- *Learning progress:* Teacher-Student CL, Graves et al., ALP-GMM, PLR, PAC, SEC, Actor-Curator.
- *Start states inside agent episodes:* Go-Explore [LIT C4], R^3 [LIT C5], failure-prefix conditioning [LIT C7], Depth Before Breadth (success-threshold frontier with resets to intermediate contexts, 2026-11-18), TRACE (prefixes predicted to give mixed outcomes), RAIL (learned bandit over anchors), Fission-GRPO's buffer of recent failures, SENTINEL and WebRL (tasks generated from failures), NoisyAgent (gap-triggered noise schedule), Bench2Robust (gated fault curriculum).
- *Regret-based UED:* PAIRED, ACCEL.
- *Correction for curriculum-induced shift in hidden aleatoric parameters:* SAMPLR (NeurIPS 2022).

**What could be new.** ActionProof's hidden commit outcomes are aleatoric parameters in SAMPLR's sense, because the ack is byte-identical. A scheduler that oversamples faulted, severe or uncertain starts raises the policy's effective p(fault | h) at the decision point. If the reward has a cost for unnecessary actions (candidate B), this teaches over-verification or over-escalation at the deployment fault rate; if verification is free, it may do no harm. Stage 3 drew half its pool from recovery starts and already logs reads per episode and escalation on clean tasks, so the shift can be measured directly. In the fetched text, the LLM-agent curricula checked (Depth Before Breadth, RAIL) apply no start-distribution correction, and a search for "curriculum-induced covariate shift" in LLM work found nothing relevant. A contribution that (1) measures the curriculum-induced change in verify, stop and escalate rates on deployment-distribution episodes under OEHF and (2) removes it, by importance-weighting start-state groups with p_deploy/p_train or by re-sampling the hidden commit outcome from the deployment posterior (SAMPLR-style), was not found for LLM tool agents.

**C is redundant if:**
- it is a priority score over start states combining failure probability, severity, uncertainty and learning progress, evaluated by training efficiency or final success. Each term and the combination pattern already exist (PLR, LILO, MoPPS, PAC, CeSoR, Depth Before Breadth, RAIL, TRACE);
- the frontier advances along scripted or reference prefixes (R^3, Reverse Curriculum, Depth Before Breadth);
- the prior shift is presented as a new phenomenon rather than as SAMPLR's CICS in a new domain;
- recovery starts are built from replayed prefixes and evaluated only on replayed prefixes. "Recovery Is a Decision" shows replay-based recovery conclusions reversing for live agents, so evaluation must include full live episodes.

### Stage 3 recovery-start curriculum (context)

As a mechanism it is not novel: Fission-GRPO, R3L, failure-prefix conditioning, ReflectPO and Depth Before Breadth all restart GRPO groups from failure-containing contexts. What remains specific to ActionProof is the information structure (silent write faults, a hidden-state reward), not the restart.

---

## 9. (c) Ranking by novelty risk (most redundant first)

1. **B: highest risk.** As specified, it is Constrained GRPO or Safety as a Budget with domain-specific indicator costs. Its only residual novelty (hidden-state-dependent costs; budgets enforced on the curriculum distribution) belongs to the setting or to C.
2. **A: high risk.** Forking, the action-class Q-minus-mean estimator, step-token application and the episode term are all published: CRR, BPO, BiPACE, RTMC, ReflectPO and ELPO in 2026, plus vine, COMA and GiGPO. Residual novelty is the OEHF analysis (hidden-label-triggered forks are biased; observation-triggered or posterior-resampled forks are not). That leans on Baisero and Amato and on SAMPLR, and would read as an application unless it is shown empirically.
3. **C: lowest risk, conditionally.** As a scored frontier scheduler it is as redundant as A. The CICS-measured-and-corrected version uses ActionProof's distinctive property directly, can be tested cheaply with Stage 3 artifacts, and has no LLM-agent precedent among the papers fetched. SAMPLR must be cited as the source of the concept.

**Suggested framing if one contribution is pursued.** Merge the residuals of A and C into one claim: "Data selection driven by privileged state (forks or start states) in observation-equivalent fault settings shifts GRPO toward over-verification; grounding the selection in the deployment posterior removes the shift." Use B's separate costs only as the instrument that makes the shift measurable (an unnecessary-action cost), not as the contribution.

---

## 10. Corrections to earlier internal notes (`docs/research/novelty_scouts_2026-11-11.md`)

- "Every agentic-RL paper with state rewards uses visible faults" is no longer accurate as written. NoisyAgent (2026-05-26) trains with misleading tool outputs, and Bench2Robust (2026-08-12) trains with silent corruptions (partial, stale, factually wrong values). The narrower claim still holds among the papers fetched: no training paper injects silent write-commit ambiguity with duplicate risk, and Bench2Robust explicitly assumes retries after explicit failures cannot duplicate.
- AgentRobustBench: confirmed that silent no-op and fabricated observation are deferred to Appendix N. Its NeurIPS 2026 and ICML 2026 workshop pages use the title "When Simulation Lies".
- Fission-GRPO is now an ACL 2026 long paper (verified at ACL Anthology).
- New close neighbours since 2026-11-11: Depth Before Breadth (2026-11-18), StateAlign (2026-11-12), Safety as a Budget (2026-11-12), TriGRPO (2026-11-02), BunshinRL (2026-11-07).

---

## UNVERIFIED (not usable)

Seen only as search snippets or summaries, not fetched; do not cite:
1. "Did It Happen? Counterfactual Evaluation of LLM Agent Recovery from Ambiguous Tool Outcomes" (Research Square rs-10730245). The fetched page showed only the title (no authors, date or abstract). A search snippet describing observation-equivalent hidden-commit timeout pairs ("Sun (2026)") probably refers to it; unconfirmed.
2. FinalityBench (arXiv 2609.04706).
3. BLINDSPOT: safety and refusal calibration in long-horizon tool agents (arXiv 2609.16305).
4. "You're Right, Let Me Fix It": agents damaging correct work when falsely accused (arXiv 2609.32616).
5. POLAR: ontology-guided risk prevention for tool-calling agents (arXiv 2610.08082).
6. AEPO, Agentic Entropy-Balanced Policy Optimization (arXiv 2510.14545).
7. "Shepherd", a tree-search RL trainer that forks at meta-agent-chosen turns (search summary only; no URL obtained).
8. CCPO, Conformal Constrained Policy Optimization for Cost-Effective LLM Agents (arXiv 2511.11828).
9. ForkGRPO for diffusion language models (arXiv 2610.19408).
10. Syllabus: Portable Curricula for RL Agents (arXiv 2411.11318).
11. Rewards for Budget-Aware Multi-Turn Agents (arXiv 2610.28409).
12. Revisiting DAgger in the Era of LLM-Agents (arXiv 2605.12913).
13. On the Hidden Objective Biases of Group-based RL (arXiv 2601.05002).
14. Reason Less, Verify More (arXiv 2607.07405).
15. AgentAbstain (arXiv 2607.10059).
16. Revisable by Design: streaming agent execution (arXiv 2604.23283).
17. Budget-Constrained Agentic LLMs (arXiv 2602.11541).
18. Detecting Silent Faults at the Environment-Trainer Boundary in RL (arXiv 2611.08144).

Withdrawn (fetched, not usable): TWICE, "Same Luck, Different Choice: Twin-World Credit Assignment for Agentic RL" (arXiv 2610.31544). Withdrawn by arXiv admin on 2026-10-26 for incorrect author affiliations. Its claimed method (paired counterfactual rollouts that share sampling noise, like CRR with common random numbers) would have been close to A.

Venue claims not verified (the papers themselves are verified above; use only the venue given in the entry):
- ARPO at ICLR 2026: an iclr.cc page appeared in search results only; the arXiv page says "working on progress".
- LILO at NeurIPS 2025: search results only; the fetched arXiv page lists no venue.
- Bench2Robust at ICLR 2026: search summary only, and inconsistent with its 2026-08-12 v1 date.
- CRR at NeurIPS 2026: arXiv comments only; a neurips.cc page appeared in search but was not fetched.
- PLR (ICML 2021), ACCEL (ICML 2022), PAIRED (NeurIPS 2020): not shown on the fetched arXiv pages.
- SPO and VinePPO proceedings pages not fetched; venues are per arXiv.
- Fission-GRPO: the arXiv abstract fetch rendered the comment as "ACM 2026 Main Conference". Treat this as a transcription error; the ACL Anthology page is authoritative.

Fetched and excluded as not relevant: Agentic Monte Carlo (arXiv 2606.05296, ICML 2026 per arXiv; black-box SMC, no training of the agent).

---

## Counts

- Verified in this session and new relative to LITERATURE.md: 84 (required items 6; candidate A 30; candidate C 22; candidate B 10; item 10 16).
- LITERATURE.md entries re-fetched in this session for method detail: 6 (Constrained GRPO, GiGPO, VinePPO, COMA, R3L, LIMBO).
- LITERATURE.md entries cited without re-fetch: 26.
- Unverified (not usable): 18 papers, 1 withdrawn paper, and venue claims for 9 otherwise verified papers (plus one transcription note for Fission-GRPO, whose venue is verified).
