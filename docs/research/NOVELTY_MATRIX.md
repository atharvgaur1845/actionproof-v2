# ActionProof novelty matrix

Compiled 2026-11-22 by a skeptical related-work pass. It extends `docs/research/NOVELTY_AUDIT.md` (84 entries, 2026-11-21) and checks the current paper (`paper/main.tex`) against the closest published work. It does not repeat the audit's fork, curriculum and constrained-optimisation entries except where the task named them.

## 0. Rules, terms and verification labels

**Terms used throughout.**
- RL: reinforcement learning. GRPO: Group Relative Policy Optimization (sample G rollouts per prompt, standardise rewards within the group, clipped policy-gradient update with a Kullback-Leibler (KL) penalty to a reference model). DAPO: Decoupled clip and Dynamic sAmpling Policy Optimization, a GRPO variant. SFT: supervised fine-tuning. DPO: Direct Preference Optimization. LoRA: low-rank adaptation.
- LLM: large language model. VLM: vision-language model. TRL: the Hugging Face Transformer Reinforcement Learning library. F1: harmonic mean of precision and recall. EMA: exponential moving average. OCR: optical character recognition. CRUD: create, read, update and delete operations. PIPE: Perturb Interface Protocol for Evaluation. ARC-DR is the method's own name in its paper (doubly-robust off-policy evaluation up to a commit point). GUI: graphical user interface. API: application programming interface. BFCL: Berkeley Function Calling Leaderboard. AUROC: area under the receiver operating characteristic curve.
- Q(s,a): expected return after action a in state s. V(s): expected return in state s. MC: Monte Carlo.
- ActionProof terms: R_V is the state reward (1 if every requested step holds in the final database). R_A is the acknowledgement reward (1 if the last acknowledgement of every requested write was ok). VSC is verified safe completion. RSC is the recovery-start curriculum. FCA is forked counterfactual action credit. A12, A14 to A17, S3 and S4 are the project's registered experiment identifiers (reward comparison, forced evidence, representation, transfer, recovery-start curriculum, registered shifts).

**Verification labels (last column of the table).**
- *Full text*: the arXiv HTML or the ACL Anthology PDF was opened in this session and the method section was read (PDFs were converted to text locally; for arXiv HTML a summarising fetch tool extracted the sections, so equations are transcriptions and must be re-copied from the source before quoting in the paper).
- *Abstract only*: the arXiv abstract page, Anthology page, proceedings page or conference page was opened; nothing beyond the abstract was checked.
- *Partly verified*: the page opened but did not show the content; content comes from a search-index snippet.
- *NOT FOUND*: no source could be located. No entry in this file has this status: every named work was located.

Every link in the table was opened in this session (2026-11-22). Entries of `NOVELTY_AUDIT.md` that were not re-opened today are not repeated here; their verification status is in that file.

---

## 1. Novelty matrix

Columns: problem; method; training signal; what the policy and the evaluator can observe ("state access"); how recovery is handled; main results; overlap with ActionProof; the difference ActionProof can defend; verification.

### 1a. Recovery training for tool agents

| Work (venue/year, link) | Problem | Method | Training signal | State access | Recovery | Main results | Overlap with ActionProof | Defensible difference | Verified how |
|---|---|---|---|---|---|---|---|---|---|
| Fission-GRPO, Zhang et al. (ACL 2026 long, pp. 40477-40491; [Anthology](https://aclanthology.org/2026.acl-long.1880/), [arXiv HTML](https://arxiv.org/html/2601.15625v2)) | Small models repeat invalid calls after execution errors | GRPO; failed rollouts become corrective prompts with simulated error diagnostics; a second "fission" group is sampled from them | Reference-call matching (function name and argument F1), format and length terms | Policy: its failed call plus explicit diagnostic text from a fine-tuned Qwen3-32B error simulator. Evaluator: reference call | Text restart from the failure; environment state is not restored; side effects ignored | Qwen3-8B BFCL v4 multi-turn 42.75% to 46.75%; error-recovery rate +5.7 points | Restarting GRPO groups from failure contexts is the RSC mechanism | Faults in ActionProof produce no error event to restart from; the reward reads authoritative state, not a reference call; RSC is not claimed as new | Full text |
| ReGRPO, Zhang and Shou (arXiv 2026; [HTML](https://arxiv.org/html/2606.31392)) | Fragile tool-using VLMs; sparse reward does not say which step failed | SFT on reflection triplets (error type, evidence, fix plan) from executed near-miss actions, then GRPO on short segments | Success, reflection-length cost, optional metadata verifier | Policy: explicit failure outputs (API errors, empty OCR). Evaluator: task success | One reflection then one corrected action; no restarts | GTA answer accuracy 67.66 vs 60.26 (SPORT); GAIA 23.35 vs 20.61 | Learning recovery from executed failures | Failures visible and local; no persistent writes; no state reward | Full text |
| PRFC, Yu, Shen and Chen (Findings of EMNLP 2026, pp. 37685-37706; [Anthology](https://aclanthology.org/2026.findings-emnlp.1762/)) | Recovery plateaus over repeated attempts; models repeat the same call error | GRPO over multi-attempt recovery trajectories with a progress-aware shaped reward | Per-attempt partial-credit call score against the reference call, plus a frontier bonus | Policy: tool execution feedback after each attempt. Evaluator: reference call | Retry within a turn, rewarded only when an attempt beats the best earlier attempt | Qwen3-4B, BFCL-v3 multi-turn +6.4 points absolute; best mean on API-Bank, T-Eval, Nestful | Recovery RL for function calling | Errors are explicit; "progress" is measured against a reference call; no hidden state, no duplicate risk | Full text (Anthology PDF) |
| ParaRecover, Guan, Yin and Shen (EMNLP 2026 main, pp. 22479-22496; [Anthology](https://aclanthology.org/2026.emnlp-main.975/)) | Final-success metrics hide whether agents localise and recover from intermediate failures in parallel tool use | Benchmark: 10,626 instances, 14 error types (planning dependency, tool selection, argument), two levels; an LLM-judge rubric (structural integrity, diagnostic reasoning, evolutionary strategy); SFT then DPO on rubric-ranked trajectories | DPO on pairs ranked by the rubric | Policy: execution graph and tool outputs. Evaluator: LLM judge plus execution metrics | Error localisation and replanning in a directed acyclic graph of calls | DPO raises Level-1 Pass@1 91.84 to 94.30; "redundant call" (an implicit, plausible-but-ineffective error) is hardest to diagnose | Process-level recovery measurement; implicit errors are the hardest | Errors are injected into trajectories, not hidden behind identical acknowledgements; no state-based reward; no RL | Full text (Anthology PDF) |
| SENTINEL, Wang et al. (arXiv 2026; [HTML](https://arxiv.org/html/2606.12908)) | Fixed task pools stop exposing the agent's current failure modes | Controller clusters failure patterns; proposer generates executable tasks targeting them; GRPO | Database-state check plus communication check, minus penalties (authentication -0.5, format -0.1, repeated identical call -0.15, length) | Policy: tool outputs (faults visible). Evaluator: final database state | New tasks, not restarts from failure states | Qwen3-4B-Thinking on tau2-bench Retail Pass^1 66.4 to 74.9 (general RL 69.0) | State reward from the database, as R_V | No silent faults; new-task generation rather than restarts; no reward-observability contrast | Full text |
| Bench2Robust, Chen et al. (arXiv 2026, v2 2026-10-11; [HTML](https://arxiv.org/html/2608.11977)) | Recovery from tool failures in realistic tool APIs | Nine failure modes (six explicit, three silent output corruptions: partial data, stale values, factual errors); five-phase gated RL curriculum; DAPO | 1.0 x efficiency x repetition if completed, else 0.3 x (matched/required actions) x repetition | Policy: tool outputs. Evaluator: task evaluator | Retry, switch tool, escalate when no path exists | RL alone +6.3 points; with an inference-time Bayesian tool memory +20.7 points | RL with silent corruptions and a recovery curriculum | Corruptions are in read outputs, not write commits. Its stated assumption: retry after an *explicit-signal* failure cannot duplicate a mutation, while retry after a silent corruption re-executes the tool. ActionProof's retries can duplicate after a byte-identical success acknowledgement | Full text |
| NoisyAgent, Chen et al. (arXiv 2026; [HTML](https://arxiv.org/html/2605.27209)) | Agents trained on clean tools fail under noisy tools and users | Gap-triggered noise curriculum (raise difficulty and share of a noise type when clean minus noisy success < 0.05); GRPO with separate normalisation for clean and noisy groups | LLM verifier (Claude Sonnet 4.5) judging final environment state | Policy: noisy tool outputs (failures, incomplete, misleading, redundant). Evaluator: LLM judge of final state | Implicit in noisy-episode training | Qwen3-32B AgentNoiseBench tau2 retail 43.20 Avg@4 | Silent misleading outputs in RL training; state-based judge | Noise perturbs observations, not write commits; judge is an LLM, not authoritative state; no acknowledgement-only reward arm | Full text |
| VeriGUI ("Don't Act Blindly"), Zhang et al. (ACL 2026 long, pp. 28924-28941; [Anthology](https://aclanthology.org/2026.acl-long.1335/), [arXiv HTML](https://arxiv.org/html/2604.05477)) | GUI agents assume actions took effect and loop on ineffective actions (72.3% of failures are timeouts from repeated ineffective actions) | Think, verify, act, predict-effect cycle; SFT with 30% synthetic failures, then GRPO | R = R_act + 0.5 R_eff + 0.5 R_ver, with R_ver = +1 correct verification, -2 hallucinated success, -0.5 missed change | Policy: next screenshot (an unchanged screen is the evidence of a silent no-op). Evaluator: reference actions and effects | Detect "no change", then correct | VeriGUI-7B recovery success 52.5% vs 45.5%; AndroidWorld 25.1% | **Training an agent to verify that an action took effect under silent no-op failures**, with a reward for correct verification | Evidence arrives free in the next observation; there is no separate read action, no partial or duplicate effect, no idempotency; the reward pays verification directly (ActionProof's R_V pays only final state) | Full text; venue verified at ACL Anthology. Note: arXiv 2508.04026, sometimes cited as "VeriGUI", now resolves to "VeriWeb", an unrelated web benchmark ([abs](https://arxiv.org/abs/2508.04026)) |
| ReflectPO, Li and Cai (arXiv 2026; [abs](https://arxiv.org/abs/2610.30933)) | Early errors cascade in long-horizon agents | Reflection-guided sampling at recovery anchors with environment snapshots; tree credit from parent-child return differences | Task reward | Stateful BFCL and tau2 | Restores snapshots at recovery anchors | Beats GRPO, GiGPO, Fission-GRPO (per audit) | Restart from recovery states in stateful APIs | No silent commits | Abstract (method from audit HTML read) |
| Depth Before Breadth, Du et al. (arXiv 2026-11-18; [abs](https://arxiv.org/abs/2611.24697)) | Long-horizon agent RL | Reverse-to-forward start-state curriculum with paired advantages | Task reward | Resets to intermediate contexts | Start-state frontier | ALFWorld 90.1/96.1, WebShop 80.2/82.6 | Start-state curricula | No hidden faults | Abstract (method from audit HTML read) |

### 1b. Algorithms named in the task (objectives)

| Work (venue/year, link) | Problem | Method | Training signal | State access | Recovery | Main results | Overlap with ActionProof | Defensible difference | Verified how |
|---|---|---|---|---|---|---|---|---|---|
| Constrained GRPO, Girgis et al. (arXiv 2026, v4 2026-09-02; [HTML](https://arxiv.org/html/2602.05863)) | Lagrangian constraints inside GRPO | Standardise reward and each cost within the group, then combine with multipliers | Indicator costs | Single-trajectory tasks | None | NAVSIM 0.774 vs 0.743; GSM8K 7B correctness 0.887 vs 0.768 (GDPO) | Separate cost terms (false completion, irreversible action) were considered and dropped | ActionProof makes no constrained-optimisation claim | Full text |
| GDPO, Liu et al. (NVIDIA report 2026; [HTML](https://arxiv.org/html/2601.05242)) | Sum-then-normalise collapses distinct reward combinations | Normalise each reward in the group, sum, then batch-normalise | Multiple rewards, including tool-calling format and correctness | n/a | None | Tool calling about +2 to 4 accuracy; AIME +6.3 on R1-1.5B | Multi-reward aggregation (ActionProof uses one non-zero reward) | None needed; not claimed | Full text |
| MO-GRPO, Ichihara et al. (TACL vol. 14, 2026; [Anthology](https://aclanthology.org/2026.tacl-1.84/)) | GRPO reward-hacks one objective in multi-objective problems | Reweight rewards by their variances | Multiple rewards | n/a | None | Balanced objectives in translation and instruction following | As GDPO | Not claimed | Abstract only |
| GiGPO, Feng et al. (NeurIPS 2025; [abs](https://arxiv.org/abs/2505.10978), [HTML](https://arxiv.org/html/2505.10978)) | Step credit in multi-turn agents | Episode advantage plus anchor-state step advantage | Outcome reward | Exact repeated environment states | None | ALFWorld >12%, WebShop >9% over GRPO | Step-level, action-conditioned credit (FCA family) | FCA is reported only as a negative result | Full text |
| COMA, Foerster et al. (arXiv 2017; venue not shown on the opened page; [abs](https://arxiv.org/abs/1705.08926)) | Multi-agent credit assignment | Centralised critic, decentralised actors, counterfactual baseline | Return | Critic: global state. Actor: own history | None | StarCraft micromanagement | FCA's all-action baseline with q = policy is COMA's baseline; same privileged-critic, history-actor split | None; the paper already says FCA is in this family | Abstract only today; equation from the audit's ar5iv read |
| IGPO, Wang et al. (ICLR 2026 poster; [ICLR page](https://iclr.cc/virtual/2026/poster/10007215), [HTML](https://arxiv.org/html/2510.14967v2)) | Sparse outcome reward in multi-turn search agents | Turn-level reward = increase in the policy's log-probability of the ground-truth answer after each turn | Information gain (needs the ground-truth answer, teacher-forced) plus outcome F1 | Policy: history. Reward: ground-truth answer | None | Average F1 60.2 vs 53.9 (DeepResearcher); +16.6 over GRPO at 3B | A dense reward for *acquiring* information; the natural alternative to R_V for teaching reads | Not compared; in ActionProof an information-gain reward would need hidden state, i.e. it is another privileged reward. Untested here | Full text |
| CIGPO, Dou (arXiv 2026; [abs](https://arxiv.org/abs/2607.16244)) | Zero-advantage lock-in for evidence-reading turns | Contextual information-gain per-turn rewards | Per-turn information gain | Read-only evidence (HotpotQA) | None | Not checked | Rewarding evidence-reading turns | Read-only; no acquisition-versus-use split | Abstract only |

### 1c. Curricula and restarts (named in the task)

| Work (venue/year, link) | Problem | Method | Training signal | State access | Recovery | Main results | Overlap | Defensible difference | Verified how |
|---|---|---|---|---|---|---|---|---|---|
| Go-Explore, Ecoffet et al. (arXiv 2019; [abs](https://arxiv.org/abs/1901.10995)) | Hard exploration | Archive of states; return to a state without exploring, then explore; robustify by imitation | Game score | Simulator restore | Return-to-state | Montezuma's Revenge >43k | Restoring to chosen states (RSC) | None for the mechanism | Abstract only |
| Reverse Curriculum Generation, Florensa et al. (CoRL 2017; [abs](https://arxiv.org/abs/1707.05300)) | Sparse goal reward | Starts near the goal, expanded backward, keeping starts of intermediate success | Sparse reward | State resets | Start-state curriculum | Solves manipulation tasks RL alone fails | Start-state curriculum | None for the mechanism | Abstract only |
| Failure-prefix conditioning, Kim et al. (arXiv 2026, v2; [HTML](https://arxiv.org/html/2601.20829)) | Saturated problems give no GRPO signal | Truncate rare wrong responses; choose the prefix whose conditioned accuracy is closest to 0.5; GRPO (Dr. GRPO variant) on prefix prompts | Binary correctness | Text prefix | Rollouts start inside a failure | Math average 44.5 vs 40.6 | RSC starts from scripted faulted prefixes | Single-turn math; no environment state | Full text |

### 1d. Evaluation of silent, ambiguous or duplicate tool outcomes (no policy training unless stated)

| Work (venue/year, link) | Problem | Method | Training signal | State access | Recovery | Main results | Overlap | Defensible difference | Verified how |
|---|---|---|---|---|---|---|---|---|---|
| "Did It Happen? Counterfactual Evaluation of LLM Agent Recovery from Ambiguous Tool Outcomes" (Research Square rs-10730245; [page](https://www.researchsquare.com/article/rs-10730245), [latest](https://www.researchsquare.com/article/rs-10730245/latest)) | A timeout after a side-effecting call does not reveal whether it executed | Every unit pairs two hidden commit states behind an identical agent-visible timeout; status affordance and stable idempotency key as controls | None (evaluation) | Agent: identical timeout. Evaluator: hidden commit state | Retry, check status | Balanced success ceiling 0.5 without an informative affordance; information alone does not guarantee correct use | **Same paired observation-equivalent design as A14 forced-evidence forks; same "information is not use" conclusion** | Training; partial and duplicate effects; reward-observability contrast | Partly verified: both pages open but show only the title; authors, date and abstract come from a search-index snippet |
| LIMBO, Li (arXiv 2026; [abs](https://arxiv.org/abs/2609.29095)) | Where exactly-once safeguards should live | Deterministic sandbox, six services, 12 fault modes, 25,930 episodes, nine models | None | Evaluator: service state | Read-back, idempotency keys | 0.5% duplicates when read-back is possible; 56-74% when not; keys cut 28% to 4%; agents report success in 90% of duplicate episodes | Duplicate effects, read-back, idempotency | Training; reward observability | Abstract only |
| UndoBench, Sah et al. (arXiv 2026; [abs](https://arxiv.org/abs/2610.05622)) | Task competence vs recovery capability | Paired trials with effect-history and state oracles | None | Evaluator: state oracles | Naive retry, idempotency, journaling | Nominal 83.54% vs conditional recovery 46.72%; naive retry duplicates in 53.33% | Recovery and duplicate effects | Training | Abstract only |
| Idempotency Is Not Recovery, Xu (arXiv 2026-11-17; [abs](https://arxiv.org/abs/2611.23353)) | Interrupted calls may have changed state without confirmation | Fault-injection study, 7,513 trials, runtime with ledgers and reconciliation | None | Runtime ledger | Reconciliation | 0/200 repeated effects vs 117/200 baseline | Duplicate suppression vs confirming commit | Training | Abstract only |
| Verified Tool Calls, Mansoor et al. (arXiv 2026; [abs](https://arxiv.org/abs/2608.02645)) | Non-atomic behaviour: timeouts after dispatch, delayed visibility, partial state updates | Wrapper: postcondition checks, verify-before-retry, idempotency keys | None (explicitly no RL) | Wrapper reads state | Verify before retry | Fewer duplicates at similar success | Partial updates and verify-before-retry | ActionProof asks whether RL *learns* this behaviour | Abstract only |
| Evidence-Guided Recovery, Zhu et al. (arXiv 2026; [abs](https://arxiv.org/abs/2610.41453)) | Ambiguous outcomes and partial execution | Planning over states consistent with evidence, effect adjudicator, completion certificate | None | Effect history | Certified recovery | 30 vs 9 certified completions over 58 paired units | Partial execution, ambiguous outcome | Training | Abstract only |
| Loud Failures, Quiet Failures, Kraishan (arXiv 2026; [abs](https://arxiv.org/abs/2610.10062)) | Detection of explicit vs well-formed-but-wrong tool outputs | Controlled fault injection | None | Evaluator | Detection | Explicit errors detected 91.3%, plausible wrong values 58.8% | Silent faults pass through when the format is normal | Write commits; training | Abstract only |
| When Tools Fail, Agents Fabricate, Kaynar et al. (NeurIPS 2026 workshop "Who Verifies the Agents?"; [page](https://neurips.cc/virtual/2026/173038)) | Silent ingestion of faulty tool data | Fault injection; verification-first monitor | None | Monitor | Corrective guidance | 19-30% confident fabrications; monitor raises recovery 42.5% to 48.0% | False success after silent faults | Training | Workshop page abstract only |
| From Confident Closing to Silent Failure, Advani (FAGEN@ICML 2026 per arXiv; [abs](https://arxiv.org/abs/2606.09863)) | False success: completion claims the state contradicts | Analysis of tau2-bench and AppWorld traces; detectors | None | Evaluator: state | n/a | False success 45-48% of failures in single-control tau2 domains; LLM judges AUROC at most 0.65 | False completion as a metric | Training | Abstract only |
| Reward Hacking Benchmark, Thaman (ICML 2026, PMLR 306; [PMLR](https://proceedings.mlr.press/v306/thaman26a.html)) | How often tool agents take shortcuts | Six exploit categories incl. skipped verification and fabricated completion markers | None | Evaluator: hidden recomputation | n/a | Exploit rates 0% to 13.9%; RL-heavy sibling 13.9% vs 0.6% | Skipped verification and false completion | Shortcuts are available by design; ActionProof's false completions come from indistinguishable acknowledgements | Abstract only |
| ToolMisuseBench, Sigdel and Baral (arXiv 2026; [HTML](https://arxiv.org/html/2604.01508v1)) | Operational tool misuse under budgets | Deterministic replayable simulators; faults: schema drift, rate limits, timeouts, authorisation failures, adversarial error rewriting | None (heuristic baselines only) | Evaluator: ground-truth state | Recovery quality, budgeted success | All baselines 0.25 success; timeout recovery 0.50 at best | Deterministic fault injection, CRUD environments | Faults are explicit; no silent commits; no training | Full text |
| AgentRobustBench / When Simulation Lies, Zhou et al. (NeurIPS 2026 Evaluations and Datasets; [abs](https://arxiv.org/abs/2605.11928)) | Sim-to-real gap for tool agents | 22 failure types; ToolRL-DR trains GRPO on uniformly perturbed data | Tool-name and parameter correctness | Policy: perturbed inputs. Evaluator: call match | One turn after explicit transient errors; silent no-op deferred to future work (per audit) | Reward-relevant and transition perturbations cut accuracy about 40% and 30% | Domain-randomised robustness training; renamed or hallucinated tool names | Silent write faults; state reward | Abstract only today (HTML read in the audit) |
| DialogTool, Wang et al. (ACL 2025 Findings per Anthology search result; [abs](https://arxiv.org/abs/2505.13328)) | Stateful tool use across a dialogue's lifecycle | 900 dialogues, 15 apps, 30 APIs, VirtualMobile environment | None | Evaluator | n/a | No model above 80% on creation and utilisation | "Stateful tool use" wording | Unrelated to commit ambiguity; evaluation only | Abstract only |
| Same Error, Different Recovery, Woo et al. (arXiv 2026-11-08; [abs](https://arxiv.org/abs/2611.11581)) | How error-message delivery changes recovery | Two pre-registered evaluation studies, 13 open models | None | n/a | n/a | Delivery format changes stopping failures | Pre-registration in agent evaluation | Errors are explicit | Abstract only |
| When Harnesses Lose the Signal, Xiao et al. (arXiv 2026; [abs](https://arxiv.org/abs/2610.00372)) | Recovery evaluated causally (with vs without intervention) | Causal intervention router, Qwen3-14B on ALFWorld | Not checked | n/a | Intervention routing | 70.33% to 73.33% | Counterfactual recovery evaluation | No silent commits | Abstract only |
| Counterfactual Evaluation in Non-Resettable Environments (ARC-DR), Bhanushali (NeurIPS 2026 workshop "Agents in the Wild"; [page](https://neurips.cc/virtual/2026/168256)) | Offline evaluation when actions are irreversible | Re-execute up to the first irreversible "commit point", doubly-robust correction after | None | Logs | n/a | Rank correlation 0.83 | Commit points of irreversible actions | Evaluation method only | Workshop page abstract only |

### 1e. Evidence use, reward observability and mechanistic work

| Work (venue/year, link) | Problem | Method | Training signal | State access | Recovery | Main results | Overlap | Defensible difference | Verified how |
|---|---|---|---|---|---|---|---|---|---|
| Proof-of-Use, Ma et al. (EMNLP 2026, pp. 37843-37857; [Anthology](https://aclanthology.org/2026.emnlp-main.1664/), [HTML v3](https://arxiv.org/html/2510.10931v3)) | RL deep-research agents make decorative tool calls that do not ground answers | GRPO with citation, perturbation-sensitivity and answer rewards | (R_cite + R_pt + R_a)/3, else -1 for bad format | Policy: retrieval outputs with stable IDs. Evaluator: citations, perturbation tests, LLM judge | Switch tools when output is empty | Unseen biomedical tool: 45.6% vs 37.1% (BioASQ) | **Acquisition without causal use under outcome RL** | Read-only tools; the paper excludes execution tools because their failures are "typically observable". ActionProof is the write-side case where they are not | Full text |
| When Your AIs Deceive You, Lang et al. (NeurIPS 2024; [abs](https://arxiv.org/abs/2402.17747)) | RL from feedback when the evaluator sees only part of the state | Formal model of deceptive inflation and overjustification | Partial-observation feedback | Evaluator partial | n/a | Conditions under which partial observation produces deceptive or over-justifying policies | **Theoretical prediction for A12**: a reward that sees only acknowledgements cannot pay for checking and pays for appearing done | ActionProof is an empirical instance with an exact information split; it adds no theory | Abstract only |
| Evaluation-Side Partially Observable Games, Li et al. (arXiv 2026-11-09; [abs](https://arxiv.org/abs/2611.12835)) | Agents deceive outcome-only evaluators | Agent, monitor and verifier game; penalties from a monitor that sees the whole trajectory | Reward plus monitor-verified penalties | Monitor: full trajectory | n/a | 26.0% better deception mitigation; up to +39.8% task success | Training under evaluators with different observability | Deception framing; no silent tool commits | Abstract only |
| Model-Adaptive Tool Necessity / Knowing-Doing Gap, Cheng et al. (NeurIPS 2026 camera ready per arXiv; [abs](https://arxiv.org/abs/2605.14038)) | Models know a tool is needed but do not call it | Probes for "necessity" and "action" | None | n/a | n/a | 26.5-54.0% mismatch (arithmetic); both signals linearly decodable, probe directions nearly orthogonal in late layers | **Same shape as A16**: information decodable, action not driven by it | ActionProof shows training moves the action margin while the probe stays at AUROC 1.00 | Abstract only |
| How Agents Know but Fail to Act (belief-action gap), Okcu et al. (NeurIPS 2026 workshop "Interpreting Agent Behavior"; [page](https://neurips.cc/virtual/2026/169059)) | State is represented more reliably than it is acted on | Belief probes and inverse RL in text grid worlds | None | n/a | n/a | Failed decisions commit later | Representation present, action not following | Grid worlds, no training contrast | Workshop page abstract only |
| LLMs Know the Constraint But Do Not Use It, Li et al. (arXiv 2026; [abs](https://arxiv.org/abs/2608.12321)) | Constraint encoded but not routed into the decision | Probes and activation interventions over 14 models | None | n/a | n/a | Probes >88%; interventions repair one failure mode (+6.4 nats), not another | Decodable but not used; nat-scale margins as in A16 | Not agents; no training | Abstract only |
| How Do Agentic LLMs Decide to Call Tools?, Gong et al. (NeurIPS 2026 per arXiv; [abs](https://arxiv.org/abs/2610.09624)) | Mechanism of the call-or-not decision | Minimal contrastive prompts; a causally mediating "tool-call vector" | None | n/a | n/a | Vector generalises across Qwen, Mistral, Granite | Causal mediation of agent actions | Shows a causal direction; ActionProof's patching did not | Abstract only |
| Interpretability without Actionability, Basu et al. (arXiv 2026; [abs](https://arxiv.org/abs/2603.18353)) | Can probes and steering fix errors? | Four intervention paradigms on medical triage | None | n/a | n/a | Probe AUROC 98.2% vs output sensitivity 45.1%; steering fixed 20%, broke 53% | Decodability does not imply controllability | Not agents | Abstract only |

### 1f. Tool-surface and interface robustness

| Work (venue/year, link) | Problem | Method | Training signal | State access | Recovery | Main results | Overlap | Defensible difference | Verified how |
|---|---|---|---|---|---|---|---|---|---|
| Can Agents Generalize to the Open World? (OpenAgent), Lv et al. (ICML 2026 per arXiv; [abs](https://arxiv.org/abs/2607.01084), [HTML](https://arxiv.org/html/2607.01084v1)) | Tool agents fail under open-world shifts | Four-level shift suite (query, action, observation, domain), including identifier perturbation, semantic conflict and reversed tool dependencies; SFT vs GRPO; perturbation-augmented SFT | SFT; GRPO with basic error codes | n/a | Refusal under fatal errors | **Both SFT and RL overfit to action order and "hallucinate the training-set topology" when dependencies change**; RL "perceives the error but still favours completion" | **Direct precedent for S4**: trained agents replay trained step sequences and fail under renamed tools | ActionProof adds multilingual requests, fault combinations and a silent-write domain; the phenomenon is not new | Full text |
| What Do Agents Learn from Trajectory-SFT: Semantics or Interfaces? (PIPE), Gu et al. (arXiv 2026; [HTML](https://arxiv.org/html/2602.01611v1)) | Do trained agents learn semantics or interface shortcuts? | Rewrite action names and formats (synonym or meaningless symbols), semantics fixed | SFT only | n/a | n/a | AgentLM-14B WebShop 61.2% to 4.44%; agents keep calling deprecated action names despite feedback; untrained agents barely move | **Precedent for S4's "training-time tool names in 13-35% of episodes"** | ActionProof shows the same after GRPO (not only SFT) | Full text |
| Hammer, Lin et al. (ICLR 2025 per search result; [abs](https://arxiv.org/abs/2410.04587)) | Function-calling models overfit to naming conventions | Training with function and parameter names masked by random strings | SFT | n/a | n/a | Masked-name tests drop other models; Hammer-7B competitive on BFCL v2 | **The standard fix for name binding**: tool-name randomisation during training | ActionProof has not tested it; it is the obvious intervention for S4 | Abstract only |
| RoTBench and RoTTuning, Ye et al. (EMNLP 2024; [Anthology](https://aclanthology.org/2024.emnlp-main.19)) | Robustness to noisy tool names and documentation | Five noise levels; training with diversified environments | SFT | n/a | n/a | GPT-4 80.00 to 58.10 under noise | Renamed-tool evaluation | As above | Abstract only |

---

## 2. Objectives of the algorithm papers (equation-level, as read)

Transcribed from the sources marked "Full text"; re-copy from the source before quoting.

- **Fission-GRPO.** Corrective context x_corr = [query; failed trajectory; diagnostic feedback f]. Fission loss J_corr(θ) = E[(1/G') Σ_j Â(τ'_j) · ratio(τ'_j | x_corr)] − β KL, with Â(τ'_j) = (R(τ'_j) − mean'_R)/(std'_R + ε) inside the fission group. Reward R(τ,t) = (1/3)[w_fmt(t) R_fmt + w_corr(t) R_corr + R_len], R_corr = α·1[function names match] + (1−α)·mean argument F1, α = 0.5; format weight decays and correctness weight grows over training.
- **ReGRPO.** R(τ) = λ_exec·1{success} − η·C(τ) + λ_val·V(x,τ), with λ_exec = 1.0, η = 0.1, λ_val = 0.3 in training; C is reflection length. Loss = −E[(1/K) Σ_k A^(k) log π(τ^(k)|x)] + β KL with A^(k) = R(τ^(k)) − mean R, K = 5, β = 0.3.
- **PRFC.** Local call score s_t ∈ [0,1] against the reference call. Frontier s̄_t = max(s_1..s_t), s̄_0 = −1. Progress reward r_t^prog = +ρ if s_t > s̄_{t−1}, else −ρ. Per-attempt reward r_t = −λ + r_t^prog + 1[t = T]·s_T, λ = ρ. Return R = Σ_t r_t; GRPO advantage A_i = (R_i − mean R)/(std R + ε); clipped trajectory-ratio objective with KL. Rollouts stop after k non-improving attempts.
- **Constrained GRPO.** max_π min_{λ≥0} J_R − Σ_k λ_k (J_Ck − d_k). Z_R,i = (R_i − mean R)/std_R, Z_Ck,i = (C_k,i − mean C_k)/std_Ck, A_i = λ_R Z_R,i − Σ_k λ_k Z_Ck,i. Multipliers are a softmax over logits updated by −(J_Ck − d_k). Theorem 4.1: scalarise-then-standardise couples every multiplier to all others through the shared variance.
- **GDPO.** A_k^(i,j) = (r_k^(i,j) − mean_j r_k)/std_j r_k per reward k; A_sum = Σ_k A_k; then batch normalisation of A_sum.
- **MO-GRPO.** Rewards reweighted by their variances before aggregation (abstract-level only).
- **GiGPO.** A^E(τ_i) = (R(τ_i) − mean R)/F_norm; A^S(a_t^i) = (R_t^i − mean over the anchor-state group)/F_norm; A = A^E + ω A^S, ω = 1, F_norm = std or 1.
- **COMA** (from the audit's ar5iv read). A^a(s,u) = Q(s,u) − Σ_{u'^a} π^a(u'^a | τ^a) Q(s,(u^{−a}, u'^a)); critic sees the global state, each actor its own history.
- **IGPO.** r_t^IG = log π(a* | q, o_≤t) − log π(a* | q, o_≤t−1), with a* the ground-truth answer (token-averaged, teacher-forced). Information-gain rewards (turns 1..T−1) and the outcome F1 (turn T) are z-normalised separately within the group; turn return R̃_t = Σ_{k≥t} γ^{k−t} r̃_k with γ = 1 replaces the GRPO advantage in a clipped objective with KL.
- **Failure-prefix conditioning.** For a saturated question q, choose the failure prefix s*(q) = argmin_s |p_θ(q,s) − τ| with τ = 0.5 (conditioned accuracy closest to one half); run Dr. GRPO on the prefix-conditioned prompts.
- **NoisyAgent.** Advance a noise type when Δ = success(clean) − success(noisy) < θ = 0.05. Â_clean = (r − mean_clean)/std_clean and Â_noise = (r − mean_noise)/std_noise, computed separately.
- **Bench2Robust.** Reward = 1.0·efficiency·repetition if completed, else 0.3·(matched/required)·repetition; efficiency = max(0.3, 1 − 0.02·max(0, |τ| − 12)). DAPO, KL 0.02, clip 0.20/0.28.
- **Proof-of-Use.** R = (R_cite + R_pt + R_a)/3 if format is valid, else −1. R_pt^t = s_t·(q' − q) with s_t = −1 for degraded cited evidence and +1 for inserted lures. R_a = α_s R_ans + (1 − α_s) R_anc, α_s = σ(κ(τ − EMA of batch correctness)).
- **VeriGUI.** R_t = R_act + α R_eff + β R_ver, α = β = 0.5; R_ver = +1 correct verdict, −2 claimed success when nothing changed, −0.5 missed change; GRPO with G = 6.
- **For comparison, ActionProof.** One binary reward (R_V or R_A), TRL GRPOTrainer defaults: group-standardised sequence advantage, token-level DAPO aggregation, sequence-level importance correction. RSC changes only the start-state distribution. FCA (dropped) adds, at a forked decision state, (Q̂(a) − V̂)/σ on the forced first-turn tokens, weighted by π_old(a|h), which is the all-action gradient Σ_a ∇π(a|h)(Q(a) − V) with forked MC estimates.

---

## 3. Scores for ActionProof's contributions

Scale 1 to 10. A standard technique applied in a new simulator is capped at 5 on conceptual and algorithmic novelty. "n/a" is used only where the dimension does not apply; such cells are not averaged.

### 3.1 The setting (ProcureSim: byte-identical acknowledgements for committed, partial, duplicate and uncommitted writes; state-only verifier; retries can duplicate)

| Dimension | Score | One-line justification |
|---|---|---|
| Conceptual novelty | 4 | Ambiguous commits, partial updates and duplicates are already the subject of LIMBO, "Did It Happen?", Verified Tool Calls, Evidence-Guided Recovery and UndoBench; only RL training inside such a setting is uncommon |
| Algorithmic novelty | 1 | No algorithm |
| Benchmark novelty | 5 | Combines partial, duplicate, never-commit and lost-acknowledgement faults with a state-only reward and paired forks; one authored domain, three workflows, sibling domains share code and authorship |
| Empirical strength | n/a | A setting, not a result |
| Mechanistic evidence | n/a | |
| Practical significance | 5 | Real failure class, but faults are far more frequent and reads cheaper than in deployment |
| Reproducibility | 7 | Deterministic simulator and released ledger; second verifier and golden traces written by language-model agents from the same specification |
| Relevance | 7 | Directly on the reliability of agents that write |

**Strongest competing explanation (for novelty):** VeriGUI already trains agents with SFT plus GRPO to verify that actions took effect under silent no-op failures, with an explicit reward for correct verification. The defensible residue is narrow: write acknowledgements that are uninformative by construction, verification that requires a separate read action, and retries that can duplicate.

### 3.2 Reward-observability finding (A12: R_V makes read-back the default, 1.00; R_A leaves the warm start unchanged, 0.00; five of five seeds)

| Dimension | Score | One-line justification |
|---|---|---|
| Conceptual novelty | 3 | Predicted by Lang et al. (an evaluator that cannot see the state cannot pay for checking it); outcome-reward agents skipping verification is documented by Reward Hacking Benchmark and Confident Closing |
| Algorithmic novelty | 1 | Two existing reward definitions |
| Benchmark novelty | 3 | The registered reward truth table is a clean instrument, not a benchmark |
| Empirical strength | 5 | Large, consistent per-seed effects on 300 untouched tasks, but one warm start, one model, and Holm-adjusted p = 0.0625 across the two primary metrics (not below 0.05); the registered mechanism prediction failed |
| Mechanistic evidence | 2 | The per-class mechanism prediction failed; the pool-composition control explains part of R_A's drift, not why it never reads |
| Practical significance | 6 | "Pay for state, not for acknowledgements" is actionable, though widely assumed |
| Reproducibility | 8 | Pre-registered, per-seed tables, deviations logged |
| Relevance | 8 | Central to the thesis |

**Strongest competing explanation:** R_A cannot reward reading by construction (reading never raises R_A), and R_A training left greedy behaviour identical to the warm start on 500 of 500 tasks, so A12 compares a trained agent with an untrained one. The R_V side may only select a behaviour the warm start already contains (30% of warm-start episodes come from a verifying oracle; Yue et al. 2025). The pending ID1 experiment (warm start that never demonstrates reading) is the test that could lift this above "expected".

### 3.3 Acquisition versus use dissociation (A14, A15, A16)

| Dimension | Score | One-line justification |
|---|---|---|
| Conceptual novelty | 4 | Proof-of-Use (read-side decorative calls), "Did It Happen?" (information alone does not guarantee use), and the knowing-doing and belief-action gap papers state the same dissociation |
| Algorithmic novelty | 1 | Forced-evidence forks are the same paired design as "Did It Happen?" |
| Benchmark novelty | 3 | Five real execution states behind one acknowledgement, with unseen partial ratios, is a careful probe set |
| Empirical strength | 5 | Clean on presence or absence (accuracy 1.00 for all models); partial repair 0.25-0.58 in four seeds and 0 in the fifth; the workflow split rests on three seeds and on small cells |
| Mechanistic evidence | 4 | Probe AUROC 1.00 everywhere including the warm start; margin gap grows under R_V; the direction ablation meets the criterion but is not R_V-specific; single-position patching is null |
| Practical significance | 6 | "Checking is not using" is a concrete deployment warning |
| Reproducibility | 7 | Registered; probe and ablation code |
| Relevance | 8 | Separates two stages that reward design treats as one |

**Strongest competing explanation:** coverage, not dissociation. Repair is learned where the training pool contains partial payments at that workflow position (the payment-first workflow) and is absent where the pool has almost none (6 of 400 RSC tasks are partial payments after a goods receipt). "Bound to workflow position" may simply be "bound to training data", which is the S4 finding again.

### 3.4 Recovery-start decomposition (S3: +0.21 VSC over R_V; 0.12 of it from a fault class R_V never trained on; start-state effect +0.09 against a class-matched control, partial payments only)

| Dimension | Score | One-line justification |
|---|---|---|
| Conceptual novelty | 3 | Restarting GRPO from failure contexts is Fission-GRPO, R3L, failure-prefix conditioning, ReflectPO, Depth Before Breadth; separating coverage from curriculum is a standard control |
| Algorithmic novelty | 1 | Known recipe, unchanged update rule |
| Benchmark novelty | 2 | Uses the setting's splits |
| Empirical strength | 5 | The +0.21 holds in five of five seeds with every RSC run above every R_V run; the start-state part rests on three seeds per arm (p = 0.05 is the floor), and one control seed learned the same repair |
| Mechanistic evidence | 2 | Per-class accounting only; no account of why recovery starts make partial repair more reliable |
| Practical significance | 5 | The honest lesson ("about half is coverage") is useful to practitioners who would credit the curriculum |
| Reproducibility | 7 | Control registered before outputs were read |
| Relevance | 7 | Tests the training lever in the thesis |

**Strongest competing explanation:** partial-payment repair is discovered or not per run (bimodal: five of five RSC runs, one of three controls), so a three-seed difference can be seed variance of a rare discovery event; recovery starts raise the discovery rate rather than teach a distinct skill. The registered extension (two more control seeds) is required before claiming the start-state effect.

### 3.5 Surface-bound procedure (S4: gains survive Hindi, Hinglish and two-fault episodes; vanish under renamed tools and an unseen four-step workflow; agents replay trained step sequences)

| Dimension | Score | One-line justification |
|---|---|---|
| Conceptual novelty | 3 | OpenAgent (ICML 2026) reports that SFT- and GRPO-trained tool agents overfit to action order and hallucinate the training topology; PIPE reports deprecated-name calls after interface rewrites; Hammer and RoTBench target name binding |
| Algorithmic novelty | 1 | Evaluation only |
| Benchmark novelty | 4 | The registered shift suite with Hindi and Hinglish requests plus two-fault episodes in a silent-write domain is less common; the renamed-tool and new-workflow shifts are standard |
| Empirical strength | 6 | Registered, five seeds per arm, Holm-controlled, clear effect sizes; a scripted oracle scores 0.91 on the unseen workflow, so the task is solvable |
| Mechanistic evidence | 2 | Behavioural evidence only (old tool names in 13-35% of episodes; skipped goods receipt in 84-96%) |
| Practical significance | 7 | The most deployment-relevant result: recovery gains did not survive an interface rename |
| Reproducibility | 8 | Frozen models, one-shot evaluation |
| Relevance | 8 | Bounds every other claim |

**Strongest competing explanation:** the training distribution, not GRPO. The warm start already calls training-time names in 62% of renamed-tool episodes, the pool has three fixed workflows and one tool vocabulary, LoRA rank is 16 and training is 200 steps. Any learner on this data would memorise the procedure. Without an intervention (for example tool-name randomisation during RL, as in Hammer, or workflow diversification), the finding cannot be attributed to the training method.

### 3.6 FCA negative result

| Dimension | Score | One-line justification |
|---|---|---|
| Conceptual novelty | 2 | All-action and counterfactual-baseline credit (COMA, TRPO vine, BiPACE, CRR, SURE-Act) |
| Algorithmic novelty | 2 | Reusing TRL's sequence-level importance weight to obtain the all-action estimator is an engineering adaptation |
| Benchmark novelty | 1 | None |
| Empirical strength | 2 | Two development seeds against a kill criterion fixed in advance; shows "did not meet the bar", not "worse in general" |
| Mechanistic evidence | 3 | Useful diagnostics: 50-60% zero-variance on-policy fork groups; forced branches carried reward differences in 66-72% of groups, yet recovery fell |
| Practical significance | 3 | A caution that forced branches cost on-policy samples |
| Reproducibility | 7 | Kill criteria fixed before training; summary script |
| Relevance | 5 | Peripheral to the thesis |

**Strongest competing explanation:** implementation trade-offs rather than the estimator: half of each decision-state group was spent on forced branches (fewer chances to discover repair), and credit went only to the first action, so a correct repair with a failed continuation was penalised. Tucker et al. 2018 already reported that action-dependent baselines rarely help in practice.

### 3.7 Process and registration (hashed pre-registrations, dated deviation log, one-shot confirmatory access, number-checked claims ledger)

| Dimension | Score | One-line justification |
|---|---|---|
| Conceptual novelty | 3 | Pre-registration is uncommon in agent RL papers but is not new in agent evaluation (Same Error, Different Recovery reports two pre-registered studies) |
| Algorithmic novelty | 1 | None |
| Benchmark novelty | 1 | None |
| Empirical strength | n/a | It is what makes the other results credible |
| Mechanistic evidence | n/a | |
| Practical significance | 5 | A template others could copy |
| Reproducibility | 9 | The strongest dimension of the project |
| Relevance | 6 | Supports, does not answer, the research question |

**Strongest competing explanation (against its value):** several deviations (a fired stop rule, a resumed crashed run, a test rule amended before outputs because the original could not be met) and a single author-run checker; reviewers may read registration as procedure rather than evidence.

---

## 4. Does the central thesis survive?

**The question as stated:** "When a tool acknowledgement is insufficient to determine whether a state-changing operation committed, how do reward observability, evidence acquisition, evidence-conditioned action and recovery training influence the reliability of an RL-trained agent?"

**Verdict: it survives as a scope, not as a thesis.** It names four factors and promises an account of each; the data give a strong answer for one (acquisition follows what the reward observes, which theory already predicts), a qualified answer for two (use is narrow; recovery training helps, about half through coverage), and a result that limits all of them (what was learned is bound to tool names and workflow order). Phrased as an open "how do ... influence" question it cannot be wrong, which is a weakness: a reviewer will ask what the claim is. It also invites comparison with OpenAgent, Proof-of-Use, "Did It Happen?" and VeriGUI, each of which already answers part of it.

**A better thesis (one testable claim that ties the results together):**

> Under silent write failures, a reward that observes authoritative state is enough to make RL-trained tool agents *acquire* evidence, but not to make them *use* it as a state-conditioned rule: what GRPO learns is a procedure keyed to workflow position and tool names. Acquisition and recovery gains carry across request language and fault combinations, and fail exactly where workflow structure or interface names change.

Why this is better: it is falsifiable with interventions the project can run (train R_V or RSC with randomised tool names in the style of Hammer and with workflow-rebalanced pools; if renamed-tool and post-receipt repair recover, the claim is supported and the cause is the training distribution, not GRPO); it turns S4 from a replication of OpenAgent into a mechanism test; and it puts A14/A15 and S4 under one explanation instead of reporting them as separate surprises. Until that intervention exists, the honest title-level claim is weaker: "acknowledged is not done, and checking is not using".

---

## 5. Threats to current paper claims (`paper/main.tex`)

1. **S4 framing is presented without its precedent.** The Discussion's "what GRPO taught here is a procedure bound to the surface it was trained on" is the main finding of OpenAgent (ICML 2026: SFT and GRPO agents overfit to action order and hallucinate the training topology) and of PIPE (agents keep calling deprecated action names after an interface rewrite). Neither is cited. Hammer and RoTBench should be cited as the known training-side remedy. Report S4 as a replication in a silent-write setting.
2. **Contribution (i) and "We study training in this setting" need narrowing.** VeriGUI (in `refs.bib` as `zhang2026verigui`, not cited in the main text) trains with SFT plus GRPO on silent no-op actions and rewards correct verification. The defensible wording is: write acknowledgements byte-identical across committed, partial, duplicate and uncommitted outcomes, where verification requires a separate read and retries can duplicate.
3. **The Bench2Robust paraphrase is slightly too broad.** The paper says it "assumes a retry cannot duplicate a write". Its text says retry after an *explicit-signal* failure does not risk a duplicate mutation, and that retry after a silent corruption re-executes the tool. Rephrase to the explicit-failure case.
4. **A16 ("training changes how represented evidence drives the action, not whether it is represented") restates a published pattern.** Cite the knowing-doing gap (NeurIPS 2026) and belief-action gap papers; present A16 as a training-time instance of it.
5. **A12 needs its theoretical anchor and its statistics stated plainly.** Cite Lang et al. 2024 for why an acknowledgement-only reward cannot teach checking. The Holm-adjusted p of 0.0625 is in the text; the abstract's "in five of five seeds" should not be read as a significance claim.
6. **Missing recovery-RL neighbours.** PRFC (Findings of EMNLP 2026) and ParaRecover (EMNLP 2026) are not cited; both are low threat but are the same-venue recovery papers a reviewer will know.
7. **"Did It Happen?" remains partly verified.** Authors and date are still not retrievable from Research Square; the footnote is accurate as written.

## 6. The five closest works

1. **"Did It Happen?"** (Research Square rs-10730245): identical observation behind two hidden commit states, and "information is not use". Same evaluation design as A14; no training. Partly verified.
2. **VeriGUI / "Don't Act Blindly"** (ACL 2026): trains agents (SFT plus GRPO) to verify action effects under silent no-op failures with a verification reward. Closest *training* precedent for the setting.
3. **OpenAgent / "Can Agents Generalize to the Open World?"** (ICML 2026 per arXiv): RL- and SFT-trained tool agents replay training-set action order and fail under identifier and dependency shifts. Closest precedent for S4.
4. **Fission-GRPO** (ACL 2026): GRPO restarted from failed tool calls. Closest precedent for RSC.
5. **Proof-of-Use** (EMNLP 2026): outcome RL produces tool calls whose results do not drive the answer. Closest precedent for acquisition without use (read-only side).

Runners-up: Bench2Robust and NoisyAgent (RL with silent output corruption), LIMBO (duplicates and read-back, evaluation only), PIPE (interface shortcutting), the knowing-doing gap paper (A16 pattern).
