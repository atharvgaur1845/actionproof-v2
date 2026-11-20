# Draft: hackathon paper (6 main pages + up to 6 appendix pages)

Working title, chosen by the registered A14 outcome (PREREGISTRATION_A14 §4), not by preference:
- if acquisition and control dissociate: *Seeing Is Not Knowing: Reward Observability and Evidence-Conditioned Control in Tool-Using Language Models*
- otherwise: *What the Reward Sees Decides What the Agent Checks*

Status markers: [PENDING file] means the number does not exist yet and may not be written until the file does. Every number in the final text resolves through `claims/claims.yaml` (`scripts/s2_numcheck.py`).

---

## Abstract (write last)

## 1. Introduction

A tool call that returns `{"status": "ok"}` tells an agent what the tool said, not what happened. In payment, procurement and other business workflows the difference matters: a write can report success without committing (a phantom write), commit only part of the requested effect, commit while its reply is lost, or fail after a timeout. When the acknowledgement is identical whether or not the write committed, only a read of the authoritative record can tell the cases apart.

Reinforcement learning is increasingly used to train such agents, and the reward has to be computed from something. A reward computed from the final database state can see what the agent cannot; a reward computed from the acknowledgements the agent received sees exactly what the agent sees. We ask two questions about this choice.

1. **Acquisition.** Holding the warm start, tasks, optimiser and reward form fixed, does the reward's information channel change whether the agent reads authoritative state after a write?
2. **Evidence-conditioned control.** When authoritative evidence is in the agent's context, does the agent's next action depend on what the evidence shows, in the correct direction?

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

**Primary result (registered H2).** On the untouched confirmatory split (`proc_conf`, 300 tasks; each frozen step-200 model evaluated once), phantom read-back is 1.00 under R_V and 0.00 under R_A in each of seeds 0, 1 and 2 (Table 2). At the task level, R_V reads back after the phantom write in all 153 tasks that contain one and R_A in none, in every seed, so the result does not depend on how many phantom attempts each arm makes. Verified completion rises by +0.313, +0.323 and +0.313. The direction is the same in every seed; with three seeds the exact sign-flip test cannot go below 1/8 (one-sided), and we claim no seed-level significance. Seeds 3 and 4 [PENDING results/s2/a12_summary.json, 5 seeds].

**Read-back is not selective.** R_V reads back after real successes almost as often as after phantom ones (observation equivalence): it has learned to read after every write, which is the only policy available when the acknowledgements are identical. R_A has stopped reading altogether, after phantom and real writes alike.

**Selection, not invention.** Decoded greedily, the warm start never reads back on the development split (0/96 phantom writes; VSC 0.495), although sampled at temperature 1 it reads back after 35% of phantom writes. GRPO under R_V turns a behaviour present only in the sampled repertoire into the greedy default; under R_A the greedy policy stays where the warm start was, and in sampled training rollouts reading falls from about 0.35 to about 0.05 within 50 steps.

**Where each reward provides gradient (exploratory; Figure 2).** Under R_V, phantom-write tasks supply 54–64% of the informative GRPO groups in every seed, and within them samples that read back earn 0.91–0.97 more reward. Under R_A, phantom tasks supply 3–19%; 68–92% come from lost-acknowledgement and never-commit tasks, where samples that read back earn 0.88–1.00 less, because acting on the read (stopping, reporting failure, or not re-calling) costs a success acknowledgement. R_A's suppression of reading is therefore driven mostly by classes where evidence changes the agent's next action, not by the phantom writes it cannot see.

**The registered mechanism test (H3) fails.** Sign agreement with the step-0 predictions is 0.6 for R_V and 0 for R_A, below the registered 0.8, so the registered composite claim C1 is not made. The test compares greedy read-back at step 200 with the greedy warm start, whose read-back is already zero, so a decrease under R_A cannot be observed; and R_V raises read-back in every class, including two predicted to fall, because it cannot tell the classes apart before reading. We report the test as registered.

**Reading is not recovery (proc_stress).** On duplicate deliveries, a fault never seen in training, R_V reads the payments after every write, sees two payments, and leaves the duplicate in place as often as R_A (duplicate rate 0.58 in both arms, every seed).

**Protocol notes.** Seed 0's R_A run fired the registered stop rule (zero-variance groups 0.864 > 0.80 at steps 40–50, reward saturated at 0.97) and was continued as a logged deviation; seeds 1–4 run under an addendum registered before they started, in which a saturated arm continues. R_V seed 2 crashed at step 199 (tokenizer error) and was resumed from its step-175 checkpoint.

## 5. Evidence-conditioned control (A14) [PENDING results/a14/a14_summary.json]

**Design.** For each of 250 fresh tasks, a scripted prefix runs to the payment acknowledgement; the branch is the real fault on that payment (committed, phantom, partial, lost acknowledgement, timeout before commit). The experimenter then forces the authoritative read of the payments into the agent's context, in the agent's own tool-call format, and the agent continues. The first decisive action (the first non-read call) is scored against the correct action for the true state (Table 3). Primary endpoint: branch-balanced correct-action accuracy on the success-acknowledgement family, BCA = ½[P(correct | phantom) + P(correct | committed)], per R_V seed; δ = 0.10, with categories evidence-conditioned, insensitive, inverted, inconclusive. Scripted references calibrate the scale (oracle 1.0; never-read and fixed-retry 0.5).

## 6. Discussion and limitations

One 3.35B model family; one environment built by the same team as the verifier; three (possibly five) training seeds, so seed-level significance is not attainable and we report direction and per-seed intervals; the acknowledgement reward is not a pure information ablation (retry payment; class mix); synthetic faults on simulated services; one annotator for the human audit.

---

## Appendix plan (≤ 6 pages)

A. Environment, tools and schemas. B. Fault taxonomy and splits (hashes). C. Full reward-truth table. D. Training configuration. E. Prompts. F. Per-seed and per-class tables. G. Statistical procedures. H. Stage 1 results (registered H1 first; unregistered contrasts labelled). I. Supporting measurements: A11 gateway factorial, H7 defences (deterministic), A13 contract coverage as finite-set results (D-A13-1). J. H4 single-annotator audit. K. Example trajectories (one per A14 branch). L. Reproduction commands, compute and HF spend. M. Deviations (D-A12-1, D-A13-1, others).
