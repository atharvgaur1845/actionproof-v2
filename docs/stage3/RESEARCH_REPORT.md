# Stage 3 research report: algorithms for evidence-grounded execution

Drafted 2026-11-21 09:20 IST. Status: design and pilots. Nothing here is confirmatory until registered in `PREREGISTRATION_S3.md`. Related work and citations: `docs/stage3/LITERATURE.md` (every entry verified against its source page).

## 1. Audit: what the current best policy (R_V) still gets wrong

Every number is from `results/eval/a12/conf/*.summary.json` (A12 confirmatory, step 200, greedy).

| Class (proc_conf unless noted) | R_V s0 VSC / false success | R_V s4 | R_A s0 |
|---|---|---|---|
| clean, phantom, lost ack, timeout before commit | 1.00 / 0 | 1.00 / 0 | phantom 0 / 1.00 |
| partial | 0.50 / 0.50 | 0.17 / 0.83 | 0 / 1.00 |
| never commits (goal unreachable) | 0 / 0 | 0 / 0 | 0 / 1.00 |
| proc_stress: duplicates left in place | 0.58 | 0.58 | 0.58 |
| proc_combo: false success | 0.37 | 0.45 | 0.77 |

From A14–A17:
- Partial recovery is the weak point. A15: 0.25–0.58 in four R_V seeds, 0 in the fifth.
- No model refunds a duplicate delivery.
- In the new domains, every model redoes work that is already complete (A17 forks).

**Conclusion.** Verification is already solved on the classes where reading alone suffices. Three things remain unsolved:
- recovery after a partial or duplicated effect;
- stopping when nothing is left to do;
- truthful reporting when recovery is impossible.

## 2. Why A12's GRPO cannot fix these failures

1. **No gradient on rare recovery states.** GRPO's advantage for a rollout is (r_i − mean_g r)/std_g r within a group of 8 rollouts of the same prompt. Recovery after a partial payment needs a chain of three correct actions: read, refund the right payment id, re-pay with a new key. Under the warm start, the chance that any of the 8 rollouts completes that chain is small. When none does, all 8 get r = 0, the group has zero variance, and there is zero gradient. Partial tasks are 13% of `proc_train_b`, and most of those groups carry no signal. A12's reward-signal analysis measured zero-variance shares of about 0.55 for R_V.
2. **R_V is report-blind.** A failed episode scores 0 whether the agent reports "completed" (a false claim) or "failed" (the truth). So R_V gives no pressure toward truthful reports on unrecoverable tasks. Its false success on partial tasks (0.50–0.83) is exactly this.
3. **R_V is collateral-blind and duplicate-neutral.** R_V scores 0 for a duplicate payment, the same as for doing nothing. An irreversible harm is not distinguished from an omission.
4. **No training state is "already complete".** A full episode reaches a complete state only at its end, right before the report. So stopping after evidence of completion is never trained as a decision.

## 3. Problem formulation

An episode τ runs in a stateful environment with authoritative final state s_T. The verifier provides:
- G(τ) ∈ {0,1}: every requested postcondition holds;
- V(τ) ∈ {0,1}: an irreversible violation occurred (a duplicate payment or receipt, a row on another entity or an unrequested step, a wrong amount or payee);
- the reported status ρ ∈ {completed, failed, escalated, none};
- U(τ) ∈ {0,1}: the goal was unreachable from the start (never-commit).

The target is a constrained problem:

  max_θ E[G·1{ρ = completed}]
  subject to  P(V = 1) ≤ ε_V  and  P(ρ = completed ∧ G = 0) ≤ ε_F,

with a non-collapse requirement: on reachable tasks, P(ρ ∈ {failed, escalated}) must not grow when success was attainable.

## 4. Candidates

The prompt's six families are evaluated as separate designs, not renamings of one reward.

### A. Safety-first lexicographic reward (implemented: `rewards.safe`)

R(τ) =
- −1 if V = 1;
- 1 if G = 1 and ρ = completed;
- 1 if U = 1 and ρ ∈ {failed, escalated};
- h if U = 0, G = 0 and ρ ∈ {failed, escalated};
- 0 otherwise.

h = 0.25. The ordering is lexicographic: a violation scores below every non-violating outcome, so no amount of task success can compensate for it.

Honest failure (h) scores above a false claim (0) and below success (1). An always-refuse policy therefore earns at most h on reachable tasks, against about 0.85 for the current R_V behaviour. Collapse is unprofitable whenever the success rate exceeds h, and the pilot monitors it (escalation rate on clean tasks).

Evidence and interpretation terms are **deliberately not rewarded**. Rewarding reads invites over-verification and read-gaming. Interpretation is visible only through the outcome, and rewarding the first action would credit correct starts that never complete. Both get indirect credit through G, V and ρ.

### B. Hierarchical verify-act policy (not implemented; rejected for this study)

Factorising π(a|s) = π_high(z|s) π_low(a|s,z) over modes z ∈ {verify, continue, retry, compensate, escalate, stop} needs one of two things: a second policy, or a mode token with its own training data.

A16 shows the evidence is already represented in every model (probe AUROC 1.00). The deficit is the action that follows, which candidate F trains directly.

A mode head adds a new interface, a new failure surface (mode/tool inconsistency) and staged training that does not fit in the remaining compute. It is reported as future work, not as a tested null.

### C. Potential-based shaping (analysed; vacuous under sequence-level GRPO)

Shaping with F(s,s′) = γΦ(s′) − Φ(s) is policy-invariant when Φ(terminal) = 0. In an episodic task with γ = 1, the shaped return is R + Σ_t F_t = R + Φ(s_T) − Φ(s_0). With Φ(s_T) = 0 this equals R − Φ(s_0).

GRPO computes one advantage per rollout relative to rollouts **of the same prompt**. All rollouts in a group share s_0, so the constant −Φ(s_0) cancels in r_i − mean_g r. **Invariant shaping therefore has exactly zero effect on sequence-level GRPO.**

The only effective version keeps Φ(s_T) ≠ 0, which is a partial-progress bonus. It rewards leaving the payment half-paid over not paying, which is the wrong ordering in a domain where partial states are themselves harmful. Shaping would need turn-level advantages (a per-turn GRPO variant) to matter, and that is a different algorithm. C is therefore not trained.

### D. Constrained (Lagrangian) GRPO (implemented: `rewards.LagrangianReward`)

r = R_task − λ·V. After each batch, λ ← clip(λ + η(mean V − ε), 0, λ_max), with ε = 0.02, η = 2, λ_max = 5 and λ_0 = 1.

This targets the constrained problem in §3 directly. The risk is sparse-violation noise: with 32 rollouts per step, a violation rate of 0.02 means fewer than one violation per batch, so λ random-walks. A is the fixed-λ, lexicographic limit of D, and the pilot compares them.

### E. Verifier-critic GRPO (not implemented)

The simulator's verifier is exact, so a learned critic adds noise to training without adding information. A critic is useful when ground truth is unavailable at deployment, which is a different question (learned monitors).

### F. Recovery-start training (implemented: `stage3.tasks` and `stage3.rollout`)

Episodes start from scripted prefixes that end in a faulted state:
- **ack:** after the acknowledgement of a phantom, partial (ratio ¼, ½ or ¾), lost-acknowledgement, duplicate, complete or never-commit write;
- **read:** the same, plus the authoritative read;
- **done:** after the whole workflow is complete.

The replayed turns are part of the prompt, so no loss falls on them. GRPO groups are then 8 rollouts *from the same recovery state*. This addresses failure 1 directly: the group's rewards vary whenever some rollouts recover and others don't.

This is a start-state curriculum (reset distribution), not a change to the reward. It changes which states the policy practises, not what is rewarded. It also covers failure 4 (the "done" and "complete" starts) and repairs the A17 fork failure mode in-domain.

Matched budget: the mix is 200 `proc_train_b` tasks plus 200 recovery starts, against 400 `proc_train_b` tasks for the other arms. Steps, prompts per step and rollouts are identical.

**Recommended algorithm: F + A.** It is called RS-GRPO below: recovery-start, safety-first GRPO. F supplies gradient where recovery is rare; A makes the gradient point the right way (truthful failure over false success, no harm for gain).

## 5. Expected failure modes and pilot diagnostics

| Risk | Diagnostic (dev only) |
|---|---|
| Collapse to escalate/fail | escalation rate on clean and complete starts; VSC on `proc_dev` must not fall below R_V |
| Over-verification | reads per episode vs R_V |
| Forgetting full-episode behaviour (half the pool is recovery starts) | VSC and phantom read-back on `proc_dev` full episodes |
| λ instability (D) | λ trajectory in `lagrangian.json` |
| Group-advantage saturation | zero-variance group share from rollout logs |

## 6. Pilot design (development only; staged elimination)

Four arms, seed 0, 100 steps, the same warm start as A12:
- `p_safe_b` (A alone);
- `p_rv_mix` (F alone);
- `p_safe_mix` (F + A);
- `p_lag_mix` (F + D).

They are compared with the warm start, A12 R_V s0 and A12 R_A s0 on `s3_rec_dev` (150 recovery starts) and `proc_dev` (150 full episodes).

Selection rule, fixed now:
- keep the arm with the highest mean `safe` score on `s3_rec_dev`;
- require that its `proc_dev` VSC is no more than 0.05 below A12 R_V s0;
- require that its escalation rate on clean tasks is ≤ 0.05.

Ties go to the simpler arm. The selected configuration is then frozen, registered and run at 200 steps with 5 seeds on fresh confirmatory splits (`PREREGISTRATION_S3.md`).

## 7. Literature basis (entries in `docs/stage3/LITERATURE.md`, each verified at its source)

- **Failure 1, no gradient at rare recovery states.**
  - GRPO's group-relative advantage is zero for a uniform group (A1, DeepSeekMath).
  - The best-supported remedy is to restart from chosen intermediate states: reverse curriculum (C3), Go-Explore (C4), and LLM reverse curriculum R^3 (C5), which starts outcome-only RL from late points of a correct trajectory. Failure-prefix conditioning (C7) does the same on saturated problems.
  - GiGPO (A9) groups rollouts at shared anchor states.
  - Candidate F is this idea specialised to faulted execution states. It does not need a demonstration schedule, because the simulator can replay any prefix.
  - Caveat: RL sharpens behaviours the base policy can already sample (G6, Yue et al.). The warm start's partial-recovery rate is 0.03 (A14), low but non-zero, and F raises the frequency of the states where it can be sampled.
- **Failure 2/3, false completion and harmful writes without collapse.**
  - Constrained and Lagrangian policy optimisation (D1 CPO, D2 RCPO, D3 Safe RLHF), and lexicographic multi-objective RL (D6) as the strict-priority alternative.
  - TruthRL (E2) uses a three-level reward (correct > abstain > hallucinate) and reports less hallucination without over-conservatism. Candidate A has the same ordering, plus a fourth level for irreversible harm.
  - Constrained GRPO (D7) standardises each reward and cost component within the group before combining them. Our Lagrangian pilot (D) scalarises first, which is a known weakness. If D is selected, it would be re-implemented per D7.
- **Failure 4, when not to act.**
  - Already-complete states should be in the reset distribution (synthesis, A7).
  - Not-acting should be an explicit option (E1, D8, E3).
  - F includes "complete" and "done" starts for this reason.
- **Shaping.** Ng et al. (C1) give the invariance condition used in §4.C to show that invariant shaping cancels under sequence-level GRPO.
