# ActionProof Stage 3: Verified Literature Review

Compiled 2026-11-21 for the ActionProof RL redesign (3.35B tool-using LM, GRPO + LoRA, simulated stateful environment with unreliable acknowledgements: phantom success, partial commit, lost ack, duplicate delivery).

## Method and verification rules

- Every entry below was verified by fetching its arXiv abstract page, ACL Anthology page, PMLR / ICLR / NeurIPS proceedings or event page, or the ICML anthology page during this session. Title, first authors, date and venue are copied from the fetched page.
- "Date" is the arXiv v1 submission date unless stated otherwise. A later version date is given when the fetched page showed one.
- "Venue" is taken from the fetched page. "(per arXiv)" means the venue appears only in the arXiv comments / journal-ref field. "(verified at ...)" means a separate proceedings or event page was fetched. When neither shows a venue, the entry says "arXiv preprint".
- OpenReview forum pages and ojs.aaai.org returned bot-check pages and were not bypassed. Venue claims that could only be checked there are listed under "UNVERIFIED".
- Algorithm descriptions are summarised from the fetched abstracts. Where a detail goes beyond the abstract (for example the exact form of a well-known objective), it is the standard published form of that method.
- Main list: 45 fully annotated entries (sections A to G). A further 18 verified papers are listed in compact form in section H.

---

## A. GRPO, its variants, zero-variance groups, multi-turn agentic RL

### A1. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- Authors: Zhihong Shao, Peiyi Wang, Qihao Zhu, et al.
- Venue: arXiv preprint (arXiv page lists no venue)
- Date: 2024-02-05 (v3 2024-04-27)
- Research question: Can math-heavy continued pretraining plus a memory-cheap PPO variant push a 7B open model toward frontier math reasoning?
- Algorithm: GRPO samples G outputs per prompt, sets each output's advantage to its reward standardised by the group mean and std (no value network), and maximises a PPO-style clipped ratio objective with a KL penalty to a reference policy.
- Main finding: DeepSeekMath 7B reaches 51.7% on MATH without tools (60.9% with self-consistency); GRPO improves reasoning while reducing PPO memory use.
- Limitation: The group-standardised advantage is exactly zero whenever all G outputs get the same reward, so always-solved and never-solved prompts produce no gradient.
- Transfer: This is our baseline and the source of problem (i): with binary outcome reward, any ActionProof state the policy always fails (for example recovery after a phantom-success ack) gives A = 0 for every sample.
- URL: https://arxiv.org/abs/2402.03300

### A2. DAPO: An Open-Source LLM Reinforcement Learning System at Scale
- Authors: Qiying Yu, Zheng Zhang, Ruofei Zhu, et al.
- Venue: NeurIPS 2025 poster (verified at https://neurips.cc/virtual/2025/poster/120129)
- Date: 2025-03-18
- Research question: Which undisclosed training details make large-scale RL for long chain-of-thought reasoning reproducible?
- Algorithm: Four techniques on a GRPO-style objective: decoupled lower and upper clip ranges (Clip-Higher), dynamic sampling that over-samples and drops prompt groups whose accuracy is 0 or 1, a token-level policy-gradient loss, and overlong reward shaping.
- Main finding: 50 points on AIME 2024 with a Qwen2.5-32B base model; training code (verl) and data released.
- Limitation: Dynamic sampling discards zero-variance groups rather than extracting signal from them, so a rare state that never yields a mixed group is never trained on.
- Transfer: Adopt dynamic sampling as the minimum fix, but pair it with state resets (section C) so rare recovery states produce mixed groups instead of being silently filtered out.
- URL: https://arxiv.org/abs/2503.14476

### A3. Understanding R1-Zero-Like Training: A Critical Perspective
- Authors: Zichen Liu, Changyu Chen, Wenjun Li, et al.
- Venue: arXiv preprint; poster at the 2nd AI for Math Workshop @ ICML 2025 (verified at https://icml.cc/virtual/2025/52393)
- Date: 2025-03-26
- Research question: How much of R1-Zero-like behaviour comes from base-model pretraining biases versus the RL optimiser?
- Algorithm: Identifies an optimisation bias in GRPO (per-response length normalisation and per-group std normalisation); Dr. GRPO removes these normalisers to obtain an unbiased objective.
- Main finding: GRPO's bias inflates the length of incorrect responses; Dr. GRPO keeps reasoning accuracy with better token efficiency (43.3% AIME 2024 with a 7B base).
- Limitation: Single-turn math only; conclusions about base models depend on the specific model families studied.
- Transfer: Drop std normalisation so near-saturated groups (1 success in 8) are not over-amplified, and drop length normalisation so long multi-call failing episodes are not under-penalised.
- URL: https://arxiv.org/abs/2503.20783

### A4. Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs
- Authors: Arash Ahmadian, Chris Cremer, Matthias Gallé, et al.
- Venue: Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers), pp. 12248-12267, August 2024 (verified at ACL Anthology)
- Date: 2024-02-22 (arXiv v1)
- Research question: Are PPO's components necessary for RLHF of LLMs, or does a simpler REINFORCE estimator work as well?
- Algorithm: REINFORCE Leave-One-Out (RLOO): draw k completions per prompt, treat each full completion as one action, and use the mean reward of the other k-1 samples as that sample's baseline.
- Main finding: REINFORCE-style variants, including RLOO, outperform PPO and "RL-free" methods such as DPO and RAFT, with fewer hyperparameters and less compute.
- Limitation: Still gives zero signal when all k rewards are equal; evaluated on single-turn preference rewards.
- Transfer: The leave-one-out baseline is an unbiased drop-in for the GRPO group mean when group size must stay small (k = 4 to 8) because multi-turn ActionProof episodes are expensive.
- URL: https://aclanthology.org/2024.acl-long.662/ (arXiv: https://arxiv.org/abs/2402.14740)

### A5. Group Sequence Policy Optimization
- Authors: Chujie Zheng, Shixuan Liu, Mingze Li, et al.
- Venue: arXiv preprint
- Date: 2025-07-24
- Research question: Does GRPO's token-level importance ratio cause instability, and is a sequence-level ratio better?
- Algorithm: GSPO defines the importance ratio from the (length-normalised) sequence likelihood and performs sequence-level clipping, rewarding and optimisation instead of per-token ratios.
- Main finding: More stable and efficient than GRPO, notably for Mixture-of-Experts models, and simplifies RL infrastructure.
- Limitation: Keeps group-relative advantages, so it does not address zero-variance groups or credit assignment across turns.
- Transfer: A turn-level or sequence-level ratio may be steadier than token-level clipping when long tool observations dominate the context; lower priority than the signal-sparsity fixes.
- URL: https://arxiv.org/abs/2507.18071

### A6. No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping
- Authors: Thanh-Long V. Le, Myeongho Jeon, Kim Vu, et al.
- Venue: ICLR 2026 (per arXiv)
- Date: 2025-09-26
- Research question: Can prompts where all rollouts get the same reward contribute learning signal instead of being discarded?
- Algorithm: RL-ZVP gives zero-variance groups a non-zero advantage whose sign follows correctness (reward all-correct, penalise all-wrong) and whose magnitude is modulated by token-level entropy.
- Main finding: Up to 8.61 accuracy points over baselines that filter such prompts, across six math benchmarks.
- Limitation: Math only; an all-wrong group is pushed down without any positive example, which is only useful if the failure is the policy's fault.
- Transfer: Gives a signal on all-fail recovery states (problem i), but in ActionProof an all-fail group can be caused by the environment fault itself, so apply it only where the simulator confirms a policy error.
- URL: https://arxiv.org/abs/2509.21880

### A7. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning
- Authors: Zihan Wang, Kangrui Wang, Qineng Wang, et al.
- Venue: arXiv preprint
- Date: 2025-04-24
- Research question: What makes multi-turn RL for LLM agents stable, and which environment and reward factors drive useful reasoning?
- Algorithm: StarPO optimises whole multi-turn trajectories; StarPO-S adds trajectory filtering that keeps high reward-variance prompts, critic incorporation, and gradient stabilisation.
- Main finding: Identifies the "Echo Trap" (a reward-variance cliff followed by gradient spikes); diverse initial states, suitable interaction granularity and fine-grained, reasoning-aware rewards are needed to avoid shallow strategies.
- Limitation: Small, symbolic environments; the collapse analysis is empirical.
- Transfer: Track within-group reward variance as an early warning of collapse to always-refuse; diverse start states supports a reset distribution that mixes faulted, clean and already-complete states.
- URL: https://arxiv.org/abs/2504.20073

### A8. ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL
- Authors: Yifei Zhou, Andrea Zanette, Jiayi Pan, et al.
- Venue: Proceedings of the 41st ICML, PMLR 235, pp. 62178-62209, 2024 (verified at PMLR)
- Date: 2024-02-29 (arXiv v1)
- Research question: How can LM agents be trained with RL over multiple turns with delayed reward, efficiently?
- Algorithm: Two RL algorithms run in parallel: an off-policy, value-based high-level critic aggregates reward over utterances (turn-level values), and a low-level token policy is trained with policy gradient using the turn-level advantage.
- Main finding: About 100x better sample efficiency than prior on-policy methods; scales to 7B models.
- Limitation: Needs a learned critic and replay buffer; critic error propagates under sparse reward.
- Transfer: A light turn-level value head would give per-tool-call advantages, isolating the call that produced a duplicate write; this conflicts with our critic-free budget, so treat as an optional add-on.
- URL: https://proceedings.mlr.press/v235/zhou24t.html (arXiv: https://arxiv.org/abs/2402.19446)

### A9. Group-in-Group Policy Optimization for LLM Agent Training
- Authors: Lang Feng, Zhenghai Xue, Tingcong Liu, et al.
- Venue: NeurIPS 2025 (per arXiv)
- Date: 2025-05-16 (latest 2025-10-28)
- Research question: Can group-based, critic-free RL provide step-level credit for long-horizon LLM agents?
- Algorithm: Episode-level advantages from trajectory groups plus step-level advantages from "anchor state" groups, built retroactively by finding identical environment states across rollouts and comparing returns of the actions taken from them.
- Main finding: More than 12% gain on ALFWorld and 9% on WebShop over GRPO-style baselines, with no extra rollout cost.
- Limitation: Requires exact repeats of environment states; noisy or continuous states rarely repeat.
- Transfer: Our simulator state (DB contents plus ack status) is hashable, so anchor groups can compare "verify" versus "retry" versus "stop" at the same post-ack state, giving step credit inside otherwise all-fail episodes.
- URL: https://arxiv.org/abs/2505.10978

### A10. Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and Credit Assignment
- Authors: Quan Wei, Siliang Zeng, Chenliang Li, et al.
- Venue: arXiv preprint (comments: "Work in progress")
- Date: 2025-05-17
- Research question: How should dense turn-level rewards be integrated into GRPO and PPO for multi-turn agents?
- Algorithm: Categorises reward structures as terminal, delayed and per-turn, and derives turn-level advantage estimators for each case for GRPO and PPO.
- Main finding: Dense per-turn rewards consistently outperform sparse terminal and delayed rewards; PPO is more stable and converges better on the QA tool tasks studied.
- Limitation: Work in progress; simple QA tool tasks with easily verifiable turn rewards.
- Transfer: Supports adding verifiable per-turn terms computed from simulator ground truth (for example a penalty on the turn that issues a second non-idempotent write) at turn level rather than folding them into the episode reward.
- URL: https://arxiv.org/abs/2505.11821

---

## B. Process versus outcome supervision, verifier-based and tool-use RL

### B1. Let's Verify Step by Step
- Authors: Hunter Lightman, Vineet Kosaraju, Yuri Burda, et al.
- Venue: ICLR 2024 (verified at proceedings.iclr.cc)
- Date: 2023-05-31 (arXiv v1)
- Research question: Is process supervision (step-level feedback) better than outcome supervision for training reliable reward models on math?
- Algorithm: Train a process reward model on human step labels (PRM800K) and score a solution by the probability that every step is correct; compare with an outcome reward model for best-of-N selection.
- Main finding: The process-supervised model solves 78% of a representative MATH subset, significantly outperforming outcome supervision.
- Limitation: Used for reranking, not policy RL; needs 800K human step labels.
- Transfer: Step labels are free in our simulator (true state after every tool call), so we get process supervision without the labelling cost.
- URL: https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html (arXiv: https://arxiv.org/abs/2305.20050)

### B2. Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
- Authors: Peiyi Wang, Lei Li, Zhihong Shao, et al.
- Venue: Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers), pp. 9426-9439, August 2024 (verified at ACL Anthology)
- Date: 2023-12-14 (arXiv v1)
- Research question: Can step-level reward labels be built automatically, and do they help verification and RL?
- Algorithm: Label each step by Monte Carlo completions from that step: hard label = whether any completion reaches the right answer, soft label = fraction that do; use the PRM for reranking and step-by-step PPO.
- Main finding: Step-by-step PPO lifts Mistral-7B from 77.9% to 84.1% on GSM8K; adding Math-Shepherd verification reaches 89.1%.
- Limitation: MC labels mix step quality with the completer's skill and cost many rollouts.
- Transfer: The rollouts we run from fork states to make GRPO groups informative also estimate a per-step value, so the same compute gives a process signal.
- URL: https://aclanthology.org/2024.acl-long.510 (arXiv: https://arxiv.org/abs/2312.08935)

### B3. Process Reinforcement through Implicit Rewards
- Authors: Ganqu Cui, Lifan Yuan, Zefan Wang, et al.
- Venue: arXiv preprint
- Date: 2025-02-03
- Research question: Can dense process rewards be used in online RL without step labels and without an expensive separately trained PRM?
- Algorithm: PRIME trains an implicit PRM with outcome labels only; per-token reward is the scaled log-ratio between the implicit PRM and a reference model; the PRM is updated online on policy rollouts and combined with outcome rewards in the advantage.
- Main finding: Eurus-2-7B-PRIME improves 15.1% on average over its starting model on reasoning benchmarks and beats Qwen2.5-Math-7B-Instruct with far less training data.
- Limitation: The implicit PRM is itself a learned model that can be exploited; token-level credit is noisy for multi-turn tool use.
- Transfer: Useful only where ground truth is missing; in ActionProof, explicit state-based step rewards are cheaper and harder to hack than a learned implicit PRM.
- URL: https://arxiv.org/abs/2502.01456

### B4. Process Reward Models for LLM Agents: Practical Framework and Directions
- Authors: Sanjiban Choudhury (sole author)
- Venue: arXiv preprint
- Date: 2025-02-14
- Research question: How can PRMs be trained and used for multi-turn LLM agents within standard RLHF pipelines?
- Algorithm: AgentPRM: an actor-critic-style loop where Monte Carlo rollouts estimate step-value targets to train a PRM and the policy is improved against it; InversePRM learns the PRM from demonstrations; discusses exploration by resetting to states an optimal policy would visit.
- Main finding: 3B models trained this way outperform GPT-4o on ALFWorld.
- Limitation: One benchmark; learned PRMs remain open to reward hacking.
- Transfer: The reset-distribution idea maps directly to fork-state resets; InversePRM suggests learning step values from oracle fault-recovery demonstrations.
- URL: https://arxiv.org/abs/2502.10325

### B5. ToolRL: Reward is All Tool Learning Needs
- Authors: Cheng Qian, Emre Can Acikgoz, Qi He, et al.
- Venue: arXiv preprint
- Date: 2025-04-16
- Research question: What reward design works best when training tool-use with RL instead of SFT?
- Algorithm: Systematic study of reward type, scale and granularity under GRPO; the proposed reward combines a format term with a fine-grained correctness term that matches tool names, parameter names and parameter values against ground truth.
- Main finding: 17% improvement over base models and 15% over SFT models.
- Limitation: Call-string matching against reference calls; no stateful side effects or unreliable tools.
- Transfer: Replace call-string matching with final-state matching against the target state, plus explicit penalties for extra or repeated writes.
- URL: https://arxiv.org/abs/2504.13958

### B6. Training Language Models to Self-Correct via Reinforcement Learning
- Authors: Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, et al.
- Venue: ICLR 2025 (verified at proceedings.iclr.cc)
- Date: 2024-09-19 (arXiv v1)
- Research question: Can an LLM learn to self-correct from its own data alone, avoiding the distribution shift and collapse of SFT-based approaches?
- Algorithm: SCoRe, two-stage multi-turn online RL: stage I trains the second attempt while constraining the first attempt to stay close to the base model; stage II optimises both attempts with a reward bonus for improvement from attempt one to attempt two.
- Main finding: State-of-the-art self-correction, improving base Gemini models by 15.6% (MATH) and 9.1% (HumanEval).
- Limitation: Two attempts only; the improvement bonus can be gamed by a deliberately weak first attempt, which stage I is designed to prevent.
- Transfer: Reward the transition from a faulted state to a repaired state with a progress bonus, while preventing the policy from causing faults to collect the bonus (use potential-based progress, C1).
- URL: https://proceedings.iclr.cc/paper_files/paper/2025/hash/871ac99fdc5282d0301934d23945ebaa-Abstract-Conference.html (arXiv: https://arxiv.org/abs/2409.12917)

---

## C. Reward shaping, credit assignment, start-state curricula and resets

### C1. Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping
- Authors: Andrew Y. Ng, Daishi Harada, Stuart Russell
- Venue: ICML 1999, pp. 278-287 (verified at ML Anthology)
- Date: 1999
- Research question: Which modifications of an MDP's reward preserve its optimal policies?
- Algorithm: Adding F(s, a, s') = gamma * Phi(s') - Phi(s) for any potential Phi over states leaves optimal policies unchanged; without knowledge of the dynamics, this is the only such class.
- Main finding: Potential-based shaping is safe; other shaping terms can create reward cycles and change optimal behaviour.
- Limitation: The guarantee is about optimal policies; with finite training and function approximation, shaping still changes learning dynamics.
- Transfer: Define Phi as verified progress of the world state (for example number of target records correctly committed and not duplicated); repeated or undone writes then net zero, whereas a bonus per "verify" call is not potential-based and invites hacking.
- URL: https://mlanthology.org/icml/1999/ng1999icml-policy/

### C2. Counterfactual Multi-Agent Policy Gradients
- Authors: Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, et al.
- Venue: arXiv preprint (arXiv page lists no venue; see UNVERIFIED for the AAAI 2018 claim)
- Date: 2017-05-24
- Research question: How to assign credit to individual agents in cooperative multi-agent RL with a shared reward?
- Algorithm: A centralised critic estimates Q(s, u); each agent's advantage is Q(s, u) minus a counterfactual baseline that marginalises that agent's action while keeping the other agents' actions fixed.
- Main finding: Improves over other multi-agent actor-critic methods on decentralised StarCraft unit micromanagement.
- Limitation: Needs a learned critic over joint actions.
- Transfer: Our simulator can replay an episode with one tool call replaced by a no-op or a verify; the outcome difference is an exact difference reward for collateral or duplicate writes, and a direct test of whether acting was needed at all.
- URL: https://arxiv.org/abs/1705.08926

### C3. Reverse Curriculum Generation for Reinforcement Learning
- Authors: Carlos Florensa, David Held, Markus Wulfmeier, et al.
- Venue: CoRL 2017 (per arXiv)
- Date: 2017-07-17
- Research question: How can goal-reaching tasks with sparse reward be learned without demonstrations or shaped rewards?
- Algorithm: Start training from states near the goal and expand the start set backward, keeping start states whose current success rate lies in an intermediate band and generating new starts by perturbing those.
- Main finding: Solves navigation and manipulation tasks that standard RL fails on.
- Limitation: Requires the ability to reset to arbitrary states and a notion of nearby states.
- Transfer: Selecting starts with intermediate success is exactly the condition for non-zero GRPO advantage; sample fork states with estimated success in roughly (0.1, 0.9) and move them earlier as the policy improves.
- URL: https://arxiv.org/abs/1707.05300

### C4. Go-Explore: a New Approach for Hard-Exploration Problems
- Authors: Adrien Ecoffet, Joost Huizinga, Joel Lehman, et al.
- Venue: arXiv preprint
- Date: 2019-01-30
- Research question: How to solve sparse or deceptive reward problems where standard exploration fails?
- Algorithm: Keep an archive of visited states, first return to a promising state without exploring (using simulator restore), then explore from it; later robustify the found trajectories with imitation learning.
- Main finding: About 43,000 mean points on Montezuma's Revenge (nearly 4x prior state of the art) and superhuman scores with domain knowledge.
- Limitation: The return step needs a resettable, ideally deterministic simulator; the state abstraction is hand-designed.
- Transfer: ActionProof is resettable: keep an archive of rare post-fault states (phantom success, partial commit, in-flight duplicate) and restore them directly as episode starts.
- URL: https://arxiv.org/abs/1901.10995

### C5. Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning
- Authors: Zhiheng Xi, Wenxiang Chen, Boyang Hong, et al.
- Venue: arXiv preprint
- Date: 2024-02-08
- Research question: Can outcome-only RL get the benefits of process supervision without step annotations?
- Algorithm: R^3 starts RL from late points in a correct demonstration (the prefix is given), then progressively slides the start point backward until the model solves the task from scratch; reward is outcome only.
- Main finding: +4.1 points on average over RL baselines on eight reasoning benchmarks; +4.2 on program-based reasoning.
- Limitation: Needs correct demonstrations and a stage schedule.
- Transfer: Use oracle trajectories through each fault type; start just after the faulty ack (where the right move is verify or repair), then move the start earlier until the policy handles the whole episode.
- URL: https://arxiv.org/abs/2402.05808

### C6. VinePPO: Refining Credit Assignment in RL Training of LLMs
- Authors: Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, et al.
- Venue: ICML 2025 (per arXiv)
- Date: 2024-10-02
- Research question: Are PPO value networks accurate for LLM reasoning, and can Monte Carlo estimates do better?
- Algorithm: Replace the value network with Monte Carlo value estimates obtained by sampling continuations from each intermediate step (possible because language environments can be resumed from any prefix).
- Main finding: Better credit assignment than PPO, with faster training and better generalisation on math benchmarks.
- Limitation: Extra rollouts per step.
- Transfer: Our simulator can resume from any step, so MC values right after a lost ack give unbiased step credit for "verify" versus "retry" without a critic.
- URL: https://arxiv.org/abs/2410.01679

### C7. Training Reasoning Models on Saturated Problems via Failure-Prefix Conditioning
- Authors: Minwu Kim, Safal Shrestha, Anubhav Shrestha, et al.
- Venue: arXiv preprint
- Date: 2026-01-28 (v2 2026-05-09)
- Research question: How can RLVR keep learning on problems the model already solves on nearly every rollout?
- Algorithm: Take prefixes of the rare incorrect trajectories on saturated problems and condition new rollouts on them, redirecting exploration to error-prone states so groups again have mixed outcomes.
- Main finding: Gains comparable to training on new medium-difficulty problems, and less degradation under misleading contexts; small trade-off on following correct early reasoning.
- Limitation: Single-turn reasoning; depends on at least some failures existing.
- Transfer: This is mechanism (i) almost exactly: when clean starts are saturated, start rollouts from failure-containing prefixes (for example right after a phantom-success ack) to recreate informative groups.
- URL: https://arxiv.org/abs/2601.20829

### C8. TreeRL: LLM Reinforcement Learning with On-Policy Tree Search
- Authors: Zhenyu Hou, Ziniu Hu, Yujiang Li, et al.
- Venue: ACL 2025 main conference (per arXiv)
- Date: 2025-06-13
- Research question: Can on-policy tree search during RL training beat independent chain sampling without a separate reward model?
- Algorithm: Build rollout trees by branching from high-uncertainty intermediate steps (not random ones); outcome rewards at leaves give process-level supervision for intermediate nodes.
- Main finding: Better than chain-based sampling on math and code under a fixed compute budget.
- Limitation: Single-turn token-level branching.
- Transfer: Branch at tool-call boundaries where the policy is uncertain (after an ambiguous ack); siblings share a prefix and form a natural GRPO group at the fork.
- URL: https://arxiv.org/abs/2506.11902

---

## D. Constrained and safe RL

### D1. Constrained Policy Optimization
- Authors: Joshua Achiam, David Held, Aviv Tamar, et al.
- Venue: ICML 2017 (per arXiv)
- Date: 2017-05-30
- Research question: Can policy search satisfy safety constraints throughout training, not only at convergence?
- Algorithm: Trust-region update that maximises reward subject to expected cost J_C <= d and a KL bound, using a new bound relating returns of two policies to their divergence.
- Main finding: Near-constraint satisfaction at every iteration on simulated robot locomotion.
- Limitation: Second-order, critic-based; expensive for LLMs.
- Transfer: Frames duplicate or collateral writes as a cost constraint with a budget d, not a reward penalty, so safety is not tuned via reward scale.
- URL: https://arxiv.org/abs/1705.10528

### D2. Reward Constrained Policy Optimization
- Authors: Chen Tessler, Daniel J. Mankowitz, Shie Mannor
- Venue: ICLR 2019 poster (per arXiv)
- Date: 2018-05-28
- Research question: How can constraints be enforced when agents exploit loopholes in a single reward?
- Algorithm: Lagrangian relaxation in which the multiplier-weighted cost acts as a penalty signal; the policy is updated on a faster timescale and the multiplier on a slower one, with convergence proofs.
- Main finding: Converges to constraint-satisfying policies in the tested domains.
- Limitation: Lagrange multipliers can oscillate; constraint is on expected cost only.
- Transfer: Easy to place inside GRPO: reward = task - lambda * violation, with lambda raised by dual ascent when the measured violation rate exceeds budget and lowered otherwise (this lowering is what stops collapse to always-refuse).
- URL: https://arxiv.org/abs/1805.11074

### D3. Safe RLHF: Safe Reinforcement Learning from Human Feedback
- Authors: Juntao Dai, Xuehai Pan, Ruiyang Sun, et al.
- Venue: ICLR 2024 (verified at proceedings.iclr.cc)
- Date: 2023-10-19 (arXiv v1)
- Research question: How to balance helpfulness and harmlessness without one objective swamping the other?
- Algorithm: Train separate reward and cost models from decoupled preferences; maximise expected reward subject to expected cost <= 0 via a Lagrangian, adapting the multiplier during fine-tuning.
- Main finding: Reduces harmful responses while improving helpfulness on Alpaca-7B compared with static trade-offs.
- Limitation: Learned cost model can be gamed; relies on preference data.
- Transfer: Keep "task completed" and "no harmful side effect" as separate signals with an adaptive multiplier; a fixed penalty either allows hacking (too small) or causes over-refusal (too large).
- URL: https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1577afd396928ed64216f3f1fd5556-Abstract-Conference.html (arXiv: https://arxiv.org/abs/2310.12773)

### D4. Optimizing the CVaR via Sampling
- Authors: Aviv Tamar, Yonatan Glassner, Shie Mannor
- Venue: AAAI 2015 (per arXiv, "To appear in AAAI 2015")
- Date: 2014-04-15
- Research question: How to optimise the Conditional Value at Risk of returns with sampled gradients?
- Algorithm: Express the CVaR gradient as a conditional expectation over the alpha-tail of outcomes and estimate it with likelihood-ratio samples, with SGD convergence guarantees.
- Main finding: A working risk-sensitive RL controller for Tetris.
- Limitation: Only the tail samples contribute, so variance is high for small alpha.
- Transfer: Optimise the worst alpha-fraction of episodes (those with ack faults) so rare catastrophic duplicate writes are not averaged away by the many clean episodes.
- URL: https://arxiv.org/abs/1404.3862

### D5. Safe Reinforcement Learning via Shielding
- Authors: Mohammed Alshiekh, Roderick Bloem, Ruediger Ehlers, et al.
- Venue: arXiv preprint (arXiv page lists no venue; see UNVERIFIED for AAAI 2018)
- Date: 2017-08-29
- Research question: How to guarantee safety during learning and execution for an RL agent?
- Algorithm: Synthesise a shield from a temporal-logic safety specification; a preemptive shield restricts available actions before the agent chooses, a post-posed shield corrects unsafe actions; convergence of the learner is preserved.
- Main finding: Safety is guaranteed while optimal policies within the safe set are still found.
- Limitation: Needs a formal specification and an abstraction of the dynamics.
- Transfer: A runtime shield can block a second non-idempotent write that has no intervening verify; log shield interventions as a cost so the policy does not learn to depend on the shield.
- URL: https://arxiv.org/abs/1708.08611

### D6. Lexicographic Multi-Objective Reinforcement Learning
- Authors: Joar Skalse, Lewis Hammond, Charlie Griffin, et al.
- Venue: IJCAI 2022 (per arXiv)
- Date: 2022-12-28 (arXiv v1)
- Research question: How to optimise several rewards with a strict priority order?
- Algorithm: Value-based and policy-gradient algorithms that optimise each objective only within a tolerance of the optimum of all higher-priority objectives; proven to converge to lexicographically optimal policies.
- Main finding: Works in practice and can encode safety constraints, compared against other constrained RL methods.
- Limitation: Tolerance choice matters; slower when many objectives are stacked.
- Transfer: Priority order (1) no unsafe or duplicate writes, (2) correct final state, (3) efficiency and no unnecessary actions; prevents trading safety for completion without fixing a penalty weight.
- URL: https://arxiv.org/abs/2212.13769

### D7. Constrained Group Relative Policy Optimization
- Authors: Roger Girgis, Rodrigue de Schaetzen, Luke Rowe, et al.
- Venue: arXiv preprint
- Date: 2026-02-05 (v4 2026-09-02)
- Research question: How should GRPO be extended to constrained optimisation with Lagrange multipliers?
- Algorithm: Indicator costs per constraint and a Lagrangian; instead of scalarising rewards and then standardising (which couples components through a shared denominator), standardise each reward and cost component within the group first, then combine them with the multipliers.
- Main finding: Better constraint satisfaction with preserved task performance on gridworld, autonomous driving and math reasoning; more reliable multiplier dynamics.
- Limitation: Preprint; not tested on tool-using agents.
- Transfer: Directly usable: compute GRPO advantages separately for task success and for each violation type (false completion claim, collateral write, duplicate write), then combine with dual-ascent multipliers.
- URL: https://arxiv.org/abs/2602.05863

### D8. Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use
- Authors: Aradhye Agarwal, Gurdit Siyan, Yash Pandya, et al.
- Venue: ICML 2026 (per arXiv)
- Date: 2026-03-03 (v2 2026-06-03)
- Research question: How to train agentic models to avoid irreversible harmful tool actions without over-refusing benign tasks?
- Algorithm: MOSAIC: a plan, check, then act-or-refuse loop with explicit safety reasoning and refusal as first-class actions, trained with preference-based RL on pairwise trajectory comparisons rather than scalar rewards.
- Main finding: Up to 50% less harmful behaviour while maintaining benign task performance across three model families.
- Limitation: Preference comparisons need a judge or annotation; harm categories differ from ack faults.
- Transfer: Make "check" (verify) and "refuse or escalate" explicit actions, and use pairwise trajectory rankings such as safe completion > justified escalation > unsafe completion to avoid both hacking and over-refusal.
- URL: https://arxiv.org/abs/2603.03205

---

## E. Hierarchical policies and abstention / when not to act

### E1. HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents
- Authors: Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, et al.
- Venue: ICML 2026 (per arXiv)
- Date: 2026-02-18 (v2 2026-05-29)
- Research question: Can separating planning from execution improve credit assignment for long-horizon, sparse-reward LLM agents?
- Algorithm: A high-level planner proposes subgoals and a low-level executor carries them out over several steps; hierarchical advantage estimation gives an unbiased gradient for both levels with provably lower variance.
- Main finding: 97.4% on ALFWorld and 83.3% on WebShop, with the largest gains on long tasks with dependent subtasks.
- Limitation: Benchmarks without unreliable tools; subgoal vocabulary is task specific.
- Transfer: Use a small high-level "mode" choice (proceed, verify, repair, no-op, escalate) above low-level tool calls; hierarchical advantages credit the mode decision separately from call formatting.
- URL: https://arxiv.org/abs/2602.16165

### E2. TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning
- Authors: Zhepei Wei, Xiao Yang, Kai Sun, et al.
- Venue: ICML 2026 (per arXiv)
- Date: 2025-09-30 (v2 2026-06-08)
- Research question: How to train models that are accurate and also abstain when they do not know, rather than guessing?
- Algorithm: GRPO with a ternary reward: positive for a correct answer, zero for abstention, negative for a hallucinated answer.
- Main finding: Hallucinations fall from 43.5% to 19.4%; truthfulness improves without the over-conservatism of baselines, attributed to better recognition of knowledge boundaries.
- Limitation: QA setting; abstention is a single terminal action.
- Transfer: Three-level episode reward: +1 verified correct completion, 0 justified escalation, -1 false completion claim or harmful write. Keeping abstention at zero (not positive) is what prevents collapse to always-refuse.
- URL: https://arxiv.org/abs/2509.25760

### E3. Acting Less is Reasoning More! Teaching Model to Act Efficiently
- Authors: Hongru Wang, Cheng Qian, Wanjun Zhong, et al.
- Venue: NeurIPS 2025 workshop poster, LAW 2025: Bridging Language, Agent, and World Models (verified at https://neurips.cc/virtual/2025/137209); arXiv v1 title was "OTC: Optimal Tool Calls via Reinforcement Learning"
- Date: 2025-04-21 (v2 2025-05-31)
- Research question: Can RL teach models to reach correct answers with fewer tool calls?
- Algorithm: OTC-PO: a tool-integrated reward that jointly scores answer correctness and tool-call efficiency, instantiated in PPO and GRPO (OTC-PPO, OTC-GRPO).
- Main finding: Up to 68.3% fewer tool calls and up to 215.4% higher tool productivity with comparable accuracy.
- Limitation: Read-only QA tools; no side effects or unreliable acknowledgements.
- Transfer: Penalise unnecessary writes in already-complete states only when the final state is correct, so efficiency pressure cannot be satisfied by doing nothing on tasks that need action.
- URL: https://arxiv.org/abs/2504.14870

---

## F. Reward hacking, tool-use reliability, side effects and duplicate writes

### F1. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
- Authors: Shunyu Yao, Noah Shinn, Pedram Razavi, et al.
- Venue: arXiv preprint (arXiv page lists no venue; see UNVERIFIED)
- Date: 2024-06-17
- Research question: How reliably do tool-using agents follow domain policies while interacting with users?
- Algorithm: Agents use domain API tools and talk to an LM-simulated user; success is judged by comparing the final database state with the annotated goal state; pass^k measures the chance that all k independent trials succeed.
- Main finding: GPT-4o succeeds on under 50% of tasks; pass^8 is below 25% in retail.
- Limitation: Simulated user adds noise; no injected tool faults.
- Transfer: Grade ActionProof on final state, not on the agent's claims, and report pass^k across randomised ack-fault seeds as the reliability metric.
- URL: https://arxiv.org/abs/2406.12045

### F2. Identifying the Risks of LM Agents with an LM-Emulated Sandbox
- Authors: Yangjun Ruan, Honghua Dong, Andrew Wang, et al.
- Venue: ICLR 2024 (verified at proceedings.iclr.cc)
- Date: 2023-09-25 (arXiv v1)
- Research question: How to find risky failures of tool-using agents at scale without building real tool sandboxes?
- Algorithm: ToolEmu: an LM emulates tool execution and an LM-based safety evaluator scores agent failures and their risk.
- Main finding: 68.8% of failures found would be valid real-world failures; even the safest agent fails about 23.9% of the time on 36 toolkits and 144 cases.
- Limitation: Emulation and grading are LM-based, so not deterministic ground truth.
- Transfer: Its risk taxonomy guides our collateral-write categories; our deterministic simulator avoids the judge noise that would otherwise become a hackable reward.
- URL: https://proceedings.iclr.cc/paper_files/paper/2024/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html (arXiv: https://arxiv.org/abs/2309.15817)

### F3. Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents
- Authors: Jiapeng Li (sole author)
- Venue: arXiv preprint (cs.LG)
- Date: 2026-09-24
- Research question: Should exactly-once behaviour under timeouts and server errors be enforced by the model, the agent harness, or the tool contract?
- Algorithm: LIMBO, a simulator of six services with twelve fault modes (including late commits, redelivery and partial batches), used to run about 25,930 episodes varying model, harness and tool contract.
- Main finding: Frontier models rarely duplicate when an acknowledgement is lost but verification is possible (0.5%), yet duplicate in 56% and 74% of episodes when the request is still in flight; idempotency keys on every write cut duplicates from 28% to 4%; harness design matters less than the contract.
- Limitation: Evaluation only, no training; single-author preprint.
- Transfer: Closest existing environment to ActionProof; its in-flight fault modes are the states where a trained verify-or-wait policy is most needed, and its contract results justify a shield or idempotency baseline to compare RL against.
- URL: https://arxiv.org/abs/2609.29095

### F4. From Confident Closing to Silent Failure: Characterizing False Success in LLM Agents
- Authors: Laksh Advani (sole author)
- Venue: FAGEN@ICML2026 workshop (per arXiv)
- Date: 2026-06-01
- Research question: How often do agents claim task completion when the environment state shows otherwise, and can judges detect it?
- Algorithm: Analysis of about 12,000 trajectories across benchmarks; "false success" is a completion claim contradicted by final state; compares LLM judges with lightweight TF-IDF detectors.
- Main finding: False-success rates range from 3% to 75.8% by domain; LLM judges reach at most AUROC 0.65, while TF-IDF detectors reach 0.83 to 0.95 and run about 3,300x faster.
- Limitation: Detection study, no training; workshop paper.
- Transfer: Do not reward completion via an LLM judge; reward and penalise completion claims only against ground-truth simulator state, which makes phantom-success claims directly punishable.
- URL: https://arxiv.org/abs/2606.09863

---

## G. Mechanistic interpretability of RL fine-tuning

### G1. Towards Best Practices of Activation Patching in Language Models: Metrics and Methods
- Authors: Fred Zhang, Neel Nanda
- Venue: ICLR 2024 (per arXiv)
- Date: 2023-09-27
- Research question: How do activation patching hyperparameters change localisation and circuit-discovery results?
- Algorithm: Systematic comparison of corruption methods (Gaussian noising versus symmetric token replacement), evaluation metrics (probability, logit difference, KL) and patching windows.
- Main finding: These choices change conclusions substantially; the paper recommends minimal-pair token replacement over Gaussian noise and logit difference as the metric.
- Limitation: Studied on standard circuits in small models.
- Transfer: Build clean and corrupt prompts that differ only in the ack content (for example a true success ack versus a phantom-success ack) and use a logit-difference metric between "verify" and "finish" tokens.
- URL: https://arxiv.org/abs/2309.16042

### G2. How to use and interpret activation patching
- Authors: Stefan Heimersheim, Neel Nanda
- Venue: arXiv preprint
- Date: 2024-04-23
- Research question: What are the practical ways to apply activation patching and how should the results be interpreted?
- Algorithm: Tutorial covering denoising (patch clean into corrupt, tests sufficiency) versus noising (patch corrupt into clean, tests necessity), metric choice, and what circuit evidence each supports.
- Main finding: Consolidated best practices; patching direction and metric determine what claim the evidence supports.
- Limitation: Guidance, not new empirical results.
- Transfer: Run both directions on the base and RL-trained checkpoints to say whether RL created a necessary verify-before-retry mechanism or amplified a pre-existing one.
- URL: https://arxiv.org/abs/2404.15255

### G3. Attribution Patching Outperforms Automated Circuit Discovery
- Authors: Aaquib Syed, Can Rager, Arthur Conmy
- Venue: NeurIPS 2023 ATTRIB Workshop (per arXiv)
- Date: 2023-10-16
- Research question: Can a linear approximation to activation patching find circuits faster and better than ACDC?
- Algorithm: Edge attribution patching (EAP): score every edge by the first-order estimate of its patching effect (activation difference times metric gradient), needing two forward passes and one backward pass, then prune the lowest-scoring edges.
- Main finding: Outperforms automated circuit discovery on circuit recovery at much lower cost.
- Limitation: Linear approximation can be poor where activations differ a lot.
- Transfer: Cheap enough to run on every saved LoRA checkpoint to track how the ack-handling circuit changes during GRPO.
- URL: https://arxiv.org/abs/2310.10348

### G4. Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms
- Authors: Michael Hanna, Sandro Pezzelle, Yonatan Belinkov
- Venue: COLM 2024 (per arXiv)
- Date: 2024-03-26
- Research question: Do scalable circuit-finding methods produce faithful circuits, and is overlap with known circuits a good test?
- Algorithm: EAP-IG integrates gradients along the path between corrupt and clean inputs instead of using a single-point gradient; circuits are evaluated by faithfulness (performance when everything outside the circuit is ablated).
- Main finding: EAP circuits are less faithful than EAP-IG circuits despite similar node overlap; faithfulness, not overlap, should be reported.
- Limitation: More compute than EAP; still approximate.
- Transfer: Report faithfulness for any claimed "ack-verification circuit" and use EAP-IG rather than EAP for final claims.
- URL: https://arxiv.org/abs/2403.17806

### G5. Reinforcement Learning Finetunes Small Subnetworks in Large Language Models
- Authors: Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, et al.
- Venue: NeurIPS 2025 (per arXiv)
- Date: 2025-05-16
- Research question: How much of an LLM does RL fine-tuning actually change?
- Algorithm: Measure parameter update sparsity after RL across 7 RL algorithms and 10 model families, with no explicit sparsity regularisation.
- Main finding: RL updates only 5% to 30% of parameters; nearly all matrices receive similarly sparse updates that are nonetheless full rank.
- Limitation: Descriptive; does not identify which behaviours the subnetwork implements.
- Transfer: Supports LoRA for RL in our setting, but the full-rank finding suggests checking whether LoRA rank limits learning of recovery behaviour; compare which modules move on clean versus faulted training.
- URL: https://arxiv.org/abs/2505.11711

### G6. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- Authors: Yang Yue, Zhiqi Chen, Rui Lu, et al.
- Venue: NeurIPS 2025 Oral (per arXiv)
- Date: 2025-04-18
- Research question: Does RLVR create new reasoning ability or re-weight what the base model can already sample?
- Algorithm: Compare pass@k curves of base and RLVR-trained models across model families and tasks for large k.
- Main finding: RLVR models win at small k, but base models reach higher pass@k at large k; RL mainly sharpens sampling toward reasoning paths already in the base distribution.
- Limitation: Mostly math and code with verifiable rewards.
- Transfer: If the base policy's pass@k on a recovery state is near zero, GRPO cannot discover recovery; seed it first (SFT on oracle recoveries or reverse-curriculum starts), then use RL to sharpen it.
- URL: https://arxiv.org/abs/2504.13837

---

## H. Additional verified papers (compact; lower priority)

All fetched and verified this session. Format: title; authors; venue; date; one-line relevance; URL.

1. REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization; Jian Hu, Jason Klein Liu, Haotian Xu, et al.; arXiv preprint; 2025-01-04 (v9 2025-11-10). Normalises advantages over the whole batch instead of per prompt group; a global baseline gives non-zero advantages even in uniform groups. https://arxiv.org/abs/2501.03262
2. Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts; Haizhong Zheng, Yang Zhou, Brian R. Bartoldson, et al.; arXiv preprint; 2025-06-02. Zero-variance prompts stay uninformative across epochs, so GRESO skips them before rollout (up to 2.4x rollout speedup); tells us which ActionProof states to stop sampling from clean starts. https://arxiv.org/abs/2506.02177
3. Selective Rollout: Mid-Trajectory Termination for Multi-Sample Agent RL; Zhiyuan Zhai, Xin Wang; arXiv preprint; 2026-05-07. About 40% of agent GRPO groups are zero-variance; stops groups whose partial action sequences have converged (prefix edit distance), 10.7% faster on ALFWorld. https://arxiv.org/abs/2605.05802
4. Tree Search for LLM Agent Reinforcement Learning; Yuxiang Ji, Ziyu Ma, Yong Wang, et al.; ICLR 2026 (per arXiv); 2025-09-25. Tree-GRPO shares prefixes across agent rollouts to get more rollouts per tool-call budget and step-wise signal from outcome rewards. https://arxiv.org/abs/2509.21240
5. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs; Jiazhan Feng, Shijue Huang, Xingwei Qu, et al.; arXiv preprint; 2025-04-15. Cold-start SFT then outcome-reward RL with interleaved code execution; shows emergent code self-correction. https://arxiv.org/abs/2504.11536
6. R³L: Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification; Weijie Shi, Yanxi Chen, Zexi Li, et al.; arXiv preprint; 2026-01-07. Restarts generation from diagnosed failure points and updates only the diverging suffix; directly matches fork-state retries. https://arxiv.org/abs/2601.03715
7. Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training; Siyu Yuan, Zehui Chen, Zhiheng Xi, et al.; arXiv preprint; 2025-01-20 (v3 2025-03-24). MCTS splices the first error step of a failed trajectory with a correct sibling path to create recovery training data. https://arxiv.org/abs/2501.11425
8. Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning (title as shown on current arXiv page); Mingyue Cheng, Shuo Yu, Daoyu Wang, et al.; arXiv preprint; 2025-11-18. Treats each interaction step as the RL transition and supports token- and step-level credit assignment. https://arxiv.org/abs/2511.14460
9. Penalizing side effects using stepwise relative reachability; Victoria Krakovna, Laurent Orseau, Ramana Kumar, et al.; arXiv preprint; 2018-06-04 (v2 2019-03-08). Side-effect penalty = choice of baseline (stepwise inaction) plus deviation measure (relative reachability); the inaction baseline is a principled reference for no-op states and collateral writes. https://arxiv.org/abs/1806.01186
10. Conservative Agency via Attainable Utility Preservation; Alexander Matt Turner, Dylan Hadfield-Menell, Prasad Tadepalli; AI, Ethics, and Society 2020 (per arXiv); 2019-02-26. Penalises changes in the ability to optimise auxiliary rewards; works even with random auxiliary rewards; a generic collateral-effect penalty. https://arxiv.org/abs/1902.09725
11. Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation; Mohamad Amin Mohamadi, Tianhao Wang, Zhiyuan Li; arXiv preprint; 2025-11-14. Ternary reward (+1, 0, -lambda) traces a Pareto frontier over risk tolerance; lambda sets how costly a wrong action is versus abstaining. https://arxiv.org/abs/2511.11500
12. TIAR: Trajectory-Informed Advantage Reweighting for LLM Abstention Learning; Muyu Pan, Shu Zhao, Nan Zhang; arXiv preprint; 2026-05-25. Uses agreement across GRPO trajectories as a confidence signal to reweight advantages for abstention. https://arxiv.org/abs/2605.25850
13. SMART: Self-Aware Agent for Tool Overuse Mitigation; Cheng Qian, Emre Can Acikgoz, Hongru Wang, et al.; Findings of ACL 2025, pp. 4604-4621 (verified at ACL Anthology); July 2025. SFT on step-level rationales for when a tool is needed; 24% fewer tool calls with better accuracy. https://aclanthology.org/2025.findings-acl.239
14. Agentic Abstention: Do Agents Know When to Stop Instead of Act?; Han Luo, Bingbing Wen, Lucy Lu Wang; NeurIPS 2026 (per arXiv); 2026-06-27 (v3 2026-11-06). Frames stopping as a sequential decision; finds agents either never abstain or abstain too late, and scale does not fix it. https://arxiv.org/abs/2606.28733
15. Agent-SafetyBench: Evaluating the Safety of LLM Agents; Zhexin Zhang, Shiyao Cui, Yida Lu, et al.; arXiv preprint; 2024-12-19. 349 environments, 2,000 cases; no agent above 60% safety; defensive prompting is insufficient. https://arxiv.org/abs/2412.14470
16. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation; Bowen Baker, Joost Huizinga, Leo Gao, et al.; arXiv preprint; 2025-03-14. CoT monitors catch reward hacking, but optimising against the monitor teaches obfuscated hacking; argues against putting a monitor score into our reward. https://arxiv.org/abs/2503.11926
17. Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures; Isham Kalappurackal Mansoor, Abhishek Phadke, Pratip Rana; arXiv preprint (cs.SE); 2026-07-31. Wrapper with postcondition checks, verify-before-retry and idempotency keys reduces duplicates under timeouts after dispatch and partial updates; a non-learned baseline for ActionProof. https://arxiv.org/abs/2608.02645
18. Failure-Transparent Agents: Benchmarking Post-Failure Reporting in Tool-Using Language Models; Junru Zhu, Shiming Xie, Aime Lu Fan Chen; ICASSP 2027 (per arXiv); 2026-09-28. False-success reporting drops from 22.8% to 0.8% with a structured evidence contract; supports requiring evidence fields before a completion claim. https://arxiv.org/abs/2609.35732

Also fetched (lower relevance, not counted in the 18 above): UndoBench (https://arxiv.org/abs/2610.05622, recovery success 46.72% vs nominal 83.54%), Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs (https://arxiv.org/abs/2509.21044, EAP shows RL strengthens and diversifies edge activations, DPO does not), Thinking Sparks!: Emergent Attention Heads in Reasoning Models During Post Training (https://arxiv.org/abs/2509.25758), Capability-Aligned Hierarchical Learning for Tool-Augmented LLMs (https://arxiv.org/abs/2606.09371), Deception Under Tool Failure in LLM Agents (WiML affinity event at NeurIPS 2026, https://neurips.cc/virtual/2026/179495; event abstract only, no arXiv link).

---

## UNVERIFIED (not usable)

Papers seen only in search-result snippets, not fetched, so not usable as citations:
- "Internalizing Agency from Reflective Experience" (arXiv 2603.16843): appeared in a search result about backtracking exploration; abstract page not fetched.
- "APPO: Procedural-Level Policy Optimization" (arXiv 2606.12384): search snippet only.
- "HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents" (arXiv 2603.00977): search snippet only.
- "Quantifying Overclaiming Propensity in Frontier LLM Agents" (arXiv 2609.20812): search snippet only.
- "Do Agents Know What They Can't Do? Evaluating Feasibility Awareness in Tool-Using Agents" (arXiv 2605.28532): search snippet only.
- "TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents" (arXiv 2602.11767): search snippet only.
- "Improving RL Exploration for LLM Reasoning through Retrospective Replay" (arXiv 2504.14363): search snippet only.

Venue claims that could not be verified (the papers themselves are verified above; use only the venue shown in the entry):
- Dr. GRPO at COLM 2025: stated by a search summary; only the ICML 2025 AI for Math Workshop poster was verified. OpenReview returned a browser-verification page.
- τ-bench at ICLR 2025: search results showed only citations inside other ICLR papers; no proceedings page for τ-bench itself was found.
- COMA (Foerster et al.) and Shielding (Alshiekh et al.) at AAAI 2018: ojs.aaai.org returned a bot-protection "Access Denied" page; not bypassed.
- Let's Verify Step by Step, SCoRe, Safe RLHF on OpenReview: OpenReview blocked by browser verification; venues were instead verified on proceedings.iclr.cc.
- TruthRL at ICML 2026 PMLR v306 and RL activation-intensity paper at ICLR 2026: seen in search results only; the TruthRL entry uses the venue from its arXiv page.

---

## Synthesis (about 300 words)

**(i) Rare recovery states get no gradient under binary outcome reward.** GRPO's advantage is non-zero only when a group has mixed outcomes (A1), so the fix is to manufacture mixed groups at the states that matter. The best-supported idea is restarting from chosen intermediate states: Go-Explore's return-then-explore (C4), Reverse Curriculum's rule of keeping starts with intermediate success (C3), and the LLM versions R^3 and failure-prefix conditioning (C5, C7), with VinePPO and AgentPRM showing that resumable language environments make this cheap (C6, B4). Second, group at shared states: GiGPO anchor states and TreeRL branching (A9, C8) turn each fork into its own group. Third, keep signal from uniform groups with DAPO filtering, RL-ZVP and Dr. GRPO normalisation (A2, A6, A3). Yue et al. (G6) adds a caveat: RL only sharpens behaviour the base policy can already sample, so recovery must be seeded first (C5, SFT).

**(ii) Preventing false completion, collateral and duplicate writes without collapse to always-refuse.** Treat safety as a constraint with an adaptive multiplier (D2, D3, D1), using Constrained GRPO's rule of standardising each component within the group before combining (D7), or a lexicographic order (D6). The multiplier falls when the constraint is met, which stops over-refusal. Ground every reward in simulator state, not claims or LLM judges: judges miss false success (F4, AUROC at most 0.65), and τ-bench grades by state (F1). Shape only with potentials over verified progress (C1), so repeated writes earn nothing. Give abstention zero reward, not a positive one (E2), and make refusal an explicit action trained against unsafe completion (D8).

**(iii) Learning when not to act.** Make no-op, verify and stop explicit high-level modes (E1, D8, A8). Penalise unnecessary actions only when the final state is correct (E3), and measure each action against a no-op counterfactual by replay (C2; H9 for the inaction baseline). Put already-complete states into the reset distribution (A7) so groups contain both acting and not-acting samples.
