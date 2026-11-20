# Draft: hackathon paper (6 main pages + up to 6 appendix pages)

Title, fixed by the registered A14 outcome (PREREGISTRATION_A14 §4: the thesis check was met, so "seeing is not knowing" is abandoned):
*What the Reward Sees Decides What the Agent Checks: Reward Observability and Evidence Use in Tool-Using Language Models*

Status markers: [PENDING file] means the number does not exist yet and may not be written until the file does. Every number in the final text resolves through `claims/claims.yaml` (`scripts/s2_numcheck.py`).

---

## Abstract

Tool-using agents are trained with rewards computed from something: the final state of the systems they change, or the replies the tools gave them. When a write can report success without committing, the two differ. We train a 3.35B Aya agent with GRPO under two rewards of identical binary form that differ only in this information channel, in a procurement environment whose success replies are byte-identical whether or not the write committed, and pre-register the comparison. On untouched tasks, the state reward yields an agent that reads the payment records after every phantom write and the acknowledgement reward one that never does, in each of five paired seeds (phantom read-back 1.00 vs 0.00; verified completion +0.27 to +0.32). Read-back was already in the warm start's sampled repertoire; the reward decides whether it is kept, and the acknowledgement reward suppresses it mainly through cases where acting on a read costs a success reply. Our pre-registered mechanism test fails, and we report it. A second pre-registered experiment forces the authoritative read into the agent's context under five real execution states: every policy, including those that never read on their own, acts correctly on whether a payment exists, so the hypothesis that reading does not produce evidence-conditioned control is rejected in this setting. Control stops where the evidence must be compared with the order: only the state-trained agents refund a partial payment, in 35–48% of tasks, and no agent removes a duplicate delivery. What the reward sees decides what the agent checks; what the agent does with evidence is largely inherited from its demonstrations.


## 1. Introduction

A tool call that returns `{"status": "ok"}` tells an agent what the tool said, not what happened. In payment, procurement and other business workflows the difference matters: a write can report success without committing (a phantom write), commit only part of the requested effect, commit while its reply is lost, or fail after a timeout. When the acknowledgement is identical whether or not the write committed, only a read of the authoritative record can tell the cases apart.

Reinforcement learning is increasingly used to train such agents, and the reward has to be computed from something. A reward computed from the final database state can see what the agent cannot; a reward computed from the acknowledgements the agent received sees exactly what the agent sees. We ask two questions about this choice.

1. **Acquisition.** Holding the warm start, tasks, optimiser and reward form fixed, does the reward's information channel change whether the agent reads authoritative state after a write?
2. **Evidence-conditioned control.** When authoritative evidence is in the agent's context, does the agent's next action depend on what the evidence shows, in the correct direction, and does the reward's channel change that?

The two are different properties. An agent can learn to issue a read after every write without its next action depending on the read's content, and an agent that never reads can still act correctly when evidence is put in front of it. We separate five stages: acquisition, interpretation, evidence-conditioned action, recovery and truthful reporting. We measure acquisition through training (A12), and evidence-conditioned action, recovery and reporting through a controlled intervention on the evidence (A14). We do not measure interpretation separately.

**Contributions.**
- A matched comparison of two GRPO rewards of identical binary form that differ only in information channel (final state vs last acknowledgement), with a reward-truth table and per-class predictions registered before training, per-seed results and a per-class analysis of where each reward provides gradient.
- A forced-evidence fork protocol: for the same task and history, the real environment is placed in a committed, absent or partial state behind a byte-identical acknowledgement, the authoritative read is put in context, and the agent continues closed-loop.
- Negative and diagnostic findings, reported as registered: read-back is selected from the warm start's repertoire, not invented by RL; the acknowledgement reward saturates and fired the registered stop rule; the registered mechanism test (H3) is reported as specified.

**Non-contributions.** The execution gateway, evidence ledger, idempotency handling and contract checks used in the demonstration are engineering, close to existing systems (LIMBO; Idempotency Is Not Recovery; Cordon); we use them as baselines and apparatus. ProcureSim is implemented in the same codebase as our verifier and is not an independent benchmark. We make no claim of exactly-once execution or real-world reliability.

## 2. Related work

**Execution integrity of tool agents.** LIMBO (arXiv 2609.29095) evaluates nine frontier models across harnesses and tool contracts on 12 boundary faults and finds that the contract explains most duplicate side effects when a read cannot resolve the fault. UndoBench (2610.05622) separates task competence from recovery across fault phases with paired trials and state oracles. Idempotency Is Not Recovery (2611.23353) and Stale Does Not Mean Unsafe (2609.29522) study runtime guards, ledgers and contract coverage. None trains the policy, and none uses success acknowledgements that are byte-identical to real success.

**Evidence use.** Proof-of-Use (EMNLP 2026) shows that deep-research agents can call tools without grounding their answers in the results and adds evidence-perturbation terms to the reward. Don't Blindly Trust It (2606.21409), oracle-state intervention (REALM 2026) and Evidence-Guided Recovery (2610.41453) show that correct information often does not correct behaviour. Loud Failures, Quiet Failures (2610.10062) finds quiet failures are detected less often and that detection does not change recovery. Our A14 is a forced-evidence intervention for stateful writes in RL-trained agents, compared across reward channels.

**Reward observability.** Lang et al. (NeurIPS 2024) show in theory that rewards computed from partial observations can produce deceptive inflation; e-PoG (2611.12835) shows RL agents exploiting what a partial-observation grader cannot see. Matched two-arm GRPO designs exist for verifier quality (Leaky Suite, 2607.11022). VeriGUI (ACL 2026) trains verification with a reward that reads ground-truth effects. We found no study that trains an LLM agent with two same-form rewards differing only in whether they read state or the agent's acknowledgements.

**RL and the warm start.** Several studies find that RL sharpens behaviours already present in the base or SFT model (Yue et al., NeurIPS 2025; Gandhi et al., 2025). Our warm start already reads back; we frame every training result as selection relative to it.

## 3. Setup

**Environment.** ProcureSim: five simulated services (order management, supplier, payments, inventory, accounting), each with its own tables and a monotonic version; reads return `as_of`. Faults sit in the write path: phantom (success reply, nothing stored), partial, lost acknowledgement (stored, timeout reply), timeout before commit (not stored, timeout reply), never commits, duplicate delivery, replica lag. Success replies are byte-identical for committed, phantom and partial writes; timeout replies are identical for lost acknowledgement and timeout before commit. Idempotency is scoped to (tool, purchase order, key); a phantom write leaves no stored key, so a same-key retry commits.

**Rewards.** R_V = 1 if every requested step is VERIFIED_SUCCESS in the final state, else 0. R_A = 1 if the last acknowledgement of every requested step's write was `ok`, else 0. Both are report-blind and binary. They differ in channel, and therefore in what they pay for: R_A pays a blind same-key retry after a lost acknowledgement (the stored success is replayed) and cannot distinguish a phantom from a commit. Table 1 (excerpt of the registered reward-truth table).

**Training.** Tiny Aya Global (3.35B) with a LoRA warm start from scripted demonstrations (30% verifying oracle, 70% acknowledgement-trusting), then GRPO: 200 steps × 4 tasks × 8 rollouts, lr 2e-5, β 0.02, temperature 1, ≤ 16 turns, on `proc_train_b` (400 tasks). Seeds 0–2 (paired: same warm start and tasks per seed index). Warm-start phantom read-back 0.35 (gate, 197/563 attempts).

**Protocol.** Pre-registrations A12 (+ addenda 1, 2) and A14, committed before the runs they cover; confirmatory splits evaluated once per frozen model (sha256), logged.

## 4. Acquisition (A12)

**Primary result (registered H2).** On the untouched confirmatory split (`proc_conf`, 300 tasks; each frozen step-200 model evaluated once), phantom read-back is 1.00 under R_V and 0.00 under R_A in each of five paired seeds (Table 2). At the task level, R_V reads back after the phantom write in all 153 tasks that contain one and R_A in none, in every seed, so the result does not depend on how many phantom attempts each arm makes. Verified completion rises by +0.313, +0.323, +0.313, +0.277 and +0.273, each per-seed interval excluding zero (descriptive hierarchical interval 0.247 to 0.355). The exact sign-flip test over seeds reaches its floor, 1/32 one-sided (2/32 two-sided); with five training runs this is the strongest seed-level evidence the design can give. Seed 0 ran under the original protocol (its R_A stop rule fired), seeds 1–4 under the addendum; results are identical in direction under both.

**Read-back is not selective.** R_V reads back after real successes almost as often as after phantom ones (observation equivalence): it has learned to read after every write, which is the only policy available when the acknowledgements are identical. R_A has stopped reading altogether, after phantom and real writes alike.

**Selection, not invention.** Decoded greedily, the warm start never reads back on the development split (0/96 phantom writes; VSC 0.495), although sampled at temperature 1 it reads back after 35% of phantom writes. GRPO under R_V turns a behaviour present only in the sampled repertoire into the greedy default; under R_A the greedy policy stays where the warm start was, and in sampled training rollouts the read rate is already 0.12–0.13 in the first 50 steps and ends at 0.04–0.06.

**Where each reward provides gradient (exploratory; Figure 2; ranges over five seeds and four 50-step blocks).** Under R_V, phantom-write tasks supply 46–75% of the informative GRPO groups, and within them samples that read back earn 0.91–0.97 more reward; the sampled read rate rises from 0.59–0.66 to 0.81–0.87. Under R_A, phantom tasks supply 3–23%; lost-acknowledgement and never-commit tasks supply 60–92%, and there samples that read back earn 0.86–1.00 less, because acting on the read (stopping, reporting failure, or not re-calling) costs a success acknowledgement; the sampled read rate falls from 0.12–0.13 to 0.04–0.06. R_A's suppression of reading is therefore driven mostly by classes where evidence changes the agent's next action, not by the phantom writes it cannot see.

**The registered mechanism test (H3) fails.** Sign agreement with the step-0 predictions is 0.6 for R_V and 0 for R_A, below the registered 0.8, so the registered composite claim C1 is not made. The test compares greedy read-back at step 200 with the greedy warm start, whose read-back is already zero, so a decrease under R_A cannot be observed; and R_V raises read-back in every class, including two predicted to fall, because it cannot tell the classes apart before reading. We report the test as registered.

**Reading is not recovery (proc_stress).** On duplicate deliveries, a fault never seen in training, R_V reads the payments after every write, sees two payments, and leaves the duplicate in place as often as R_A (duplicate rate 0.58 in both arms, every seed).

**Protocol notes.** Seed 0's R_A run fired the registered stop rule (zero-variance groups 0.864 > 0.80 at steps 40–50, reward saturated at 0.97) and was continued as a logged deviation; seeds 1–4 run under an addendum registered before they started, in which a saturated arm continues. R_V seed 2 crashed at step 199 (tokenizer error) and was resumed from its step-175 checkpoint.

## 5. Evidence-conditioned control (A14)

**Design.** For each of 250 fresh tasks, a scripted prefix runs to the payment acknowledgement; the branch is the real fault on that payment (committed, phantom, partial, lost acknowledgement, timeout before commit). The experimenter then forces the authoritative read of the payments into the agent's context, in the agent's own tool-call format, and the agent continues closed-loop. The first decisive action (the first non-read call) is scored against the correct action for the true state (Table 3). Primary endpoint: branch-balanced correct-action accuracy on the success-acknowledgement family, BCA = ½[P(correct | phantom) + P(correct | committed)], per R_V seed; δ = 0.10. Scripted references calibrate the scale (oracle 1.00; never-read and fixed-retry 0.50 by construction). The read is an intervention, so A14 measures what a policy does with evidence it is given, not whether it seeks it.

**Result (registered).** Every policy, including the warm start and every R_A seed, which never reads on its own, re-pays when the read shows no payment and proceeds when it shows the payment: BCA 1.00 [1.00, 1.00] and signed S +1.00 for all of them, and the same on the timeout family. All registered contrasts are equivalent within δ. The registered thesis check was met, so we abandon the hypothesis that reading does not by itself produce evidence-conditioned control: in this setting, once the evidence is in context, the next action follows it.

**Where control stops.** On the partial branch the read shows one payment for less than the purchase order, and the correct action is a refund (or escalation). R_V seeds choose it in 0.36, 0.48 and 0.35 of tasks; every R_A seed in none; the warm start in 0.03. Acting on whether a row exists is in the warm start's repertoire; acting on what a row says, when that requires comparing an amount with the order, is mostly not, and only the state reward moves it. The same limit appears in the confirmatory stress split: R_V reads the payments after a duplicate delivery and leaves the duplicate in place as often as R_A.

**Interface ablation (exploratory).** When the same read is attributed to the runtime instead of the agent's own call, R_V stays at BCA 0.99–1.00 but R_A falls to 0.83 (S +0.66) in every seed, and partial-branch accuracy collapses for R_V (0.01–0.14). Reading the trajectories suggests why: after the forced read, the warm start continues in the style of its verifying demonstrator, reading back after every later write. The control measured by A14 is largely inherited from the verifying demonstrations and depends on the evidence looking like them.

**Taken together.** The reward's channel decides whether the agent acquires evidence (A12); the agent's response to evidence it has is mostly fixed by its demonstrations, already correct for presence and absence, weak for content that must be compared, and improved only under the state reward. For practice, this suggests putting evidence in front of the agent (as the gateway's Enforce+Inform mode does; A11: +0.25 verified completion for the warm start) is more reliable than training it to look.

## 6. Discussion and limitations

**What the evidence supports.** A reward's information channel selects evidence acquisition in a controlled comparison, consistently over five training seeds; the effect is selection from the warm start's repertoire, not invention, and it is not selective (R_V reads after every write). Given evidence, agents act on simple presence or absence regardless of how they were trained; acting on evidence that must be compared is rare and improves only under the state reward. A practical reading: a runtime that verifies writes and shows the verified state to the agent (A11) does more for reliability than training the agent to look.

**What it does not support.** Generality beyond one 3.35B model family, one environment built by the same team as the verifier, synthetic faults and five training seeds. The acknowledgement reward is not a pure information ablation: it pays a blind same-key retry after a lost acknowledgement, and most of its gradient comes from classes where acting on a read costs a reply; the phantom-write effect is partly true by construction (R_A cannot see a phantom). The registered mechanism test compared greedy policies against a greedy warm start that never reads, so it could not observe suppression. The forced-evidence result may depend on the evidence resembling the verifying demonstrations (the runtime-attributed ablation lowers R_A to 0.83). The human audit (H4) has one annotator.

**With more credits.** (1) A second model family and an environment not built by us (an existing tool benchmark with injected phantom acknowledgements), same registered design; expected: the channel effect on acquisition replicates where the warm start samples read-back at all. (2) A reward contrast without the retry-payment confound (an acknowledgement reward that ignores idempotency-store replies), to isolate information from incentive. (3) Training that targets content-conditioned control (partial and duplicate payments in the training mix with the state reward); expected: partial-branch accuracy above the 0.35–0.48 reached here.

---

## Appendix plan (≤ 6 pages)

A. Environment, tools and schemas. B. Fault taxonomy and splits (hashes). C. Full reward-truth table. D. Training configuration. E. Prompts. F. Per-seed and per-class tables. G. Statistical procedures. H. Stage 1 results (registered H1 first; unregistered contrasts labelled). I. Supporting measurements: A11 gateway factorial, H7 defences (deterministic), A13 contract coverage as finite-set results (D-A13-1). J. H4 single-annotator audit. K. Example trajectories (one per A14 branch). L. Reproduction commands, compute and HF spend. M. Deviations (D-A12-1, D-A13-1, others).


## Declaration of prior work

ActionProof Stage 1 (Nov 1–15, 2026; submitted Nov 15) built the e-commerce environment, the PCSTR rewards, the Stage 1 GRPO runs and the judge audit; those results are reported as Stage 1 and are not part of this paper's claims except where labelled. Stage 2 (Nov 20–25) built ProcureSim, the R_V/R_A rewards, the gateway, the A11–A14 pre-registrations, all Stage 2 training and evaluation, and this paper. The Aya models are Cohere's; the toolkits used (AgentTune for GRPO rollouts, AuditKIT for the judge) are the hackathon's. Claude (Anthropic) was used as a coding and writing assistant throughout; every number was produced by code in the repository and checked against its file.
