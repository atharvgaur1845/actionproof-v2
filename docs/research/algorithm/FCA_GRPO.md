# Forked counterfactual action credit for GRPO (FCA)

Status: design, 2026-11-21. Novelty is assessed in `docs/research/NOVELTY_AUDIT.md`. This note states exactly what changes relative to the trainer actually in use, TRL 1.7.1 `GRPOTrainer` through AgentTune `TrlAgenticGrpo`, read from source on the cluster.

## 0. The baseline update rule actually in use

The Stage 2 and Stage 3 runs use these settings, all TRL defaults unless noted:
- `loss_type = "dapo"`: the token-level clipped surrogate, summed over masked tokens and divided by the number of masked tokens in the batch;
- `scale_rewards = "group"`;
- `multi_objective_aggregation = "sum_then_normalize"`;
- `num_iterations = 1`, ε = 0.2;
- `beta = 0.02`: KL to the reference, the k3 estimator, per token;
- `vllm_importance_sampling_correction = True`, `vllm_importance_sampling_mode = "sequence_mask"`, `clip_max = 3`, `clip_min = None`;
- the multi-turn `env_mask`, which sets the loss mask to 0 on template and tool-result tokens.

For a group of G completions of one prompt with rewards r_1..r_G (only the R_V reward has non-zero weight), each completion gets one sequence-level advantage:

  Â_i = (r_i − mean_g r) / (std_g r + 1e-4)

and every policy token t of completion i is weighted by:

  ρ_i = exp(Σ_t m_it (log π_old(y_it) − log μ(y_it))),

where μ is vLLM's sampling probability. ρ_i is set to 0 if ρ_i > 3.

The per-token loss is −min(c_it Â_i, clip(c_it) Â_i)·ρ_i + β KL_it, with c_it = π_θ(y_it)/π_old(y_it). Strictly, this is DAPO-style token aggregation with group-relative advantages; the paper should say so.

## 1. Setting

- **Simulator state.** s ∈ S is the full database (hidden from the policy).
- **Policy context.** h is the conversation: system prompt, request, tool calls and tool results. The policy observes h only.
- **Actions.** a is one assistant turn (tool calls as JSON).
- **Episode reward.** R_V(τ) = 1 if every requested postcondition holds in the final state s_T, else 0. The verifier reads s_T; the policy never does.
- **Decision state.** A *decision state* (s*, h*) is the end of a recovery-start prefix (Stage 3 tasks, `meta.start_prefix`): the state right after a faulted write's acknowledgement (`ack` starts) or after its authoritative read (`read` starts).

At a decision state, the next action falls into a small set of **action classes** K(h*), defined from h* alone:

| Class | `read` start (evidence in h*) | `ack` start (no evidence yet) |
|---|---|---|
| proceed | the next requested step's write, or `submit_report(completed)` if none | same |
| retry | the same write with the same idempotency key | same |
| compensate | `refund_payment(<last live payment id in the read>)` for pay; record the missing quantity for receive; available only if the read shows a payment or receipt | not available |
| escalate | `escalate_to_human(po, reason)` | same |
| read | not used (already read) | the authoritative read for the step |

The arguments of every forced action are computed from h* only: the PO and amounts from the request, ids from tool results the policy has seen. No hidden state enters a forced action or any prompt (tested).

## 2. Method

For a group whose prompt is a decision-state task, the G = 8 slots are split:
- **M = 4 on-policy rollouts** from h*;
- **G − M = 4 forced branches**, one per class in K(h*) (cycled if |K| < 4). Each forced branch appends the canonical text of its class as the first assistant turn, then continues with on-policy sampling to the end of the episode.

Every branch runs in its own copy of the simulator, rebuilt from the same task and the same scripted prefix. Prompts without a decision state (full episodes) keep G = 8 on-policy rollouts and the unchanged TRL advantage.

**Estimates.** With D_a the set of completions whose first decisive action is class a (forced branch a, plus on-policy rollouts whose first non-read action falls in class a):

  Q̂(a) = mean_{j ∈ D_a} r_j
  V̂ = mean_{i on-policy} r_i      (an unbiased estimate of V^π(h*))
  σ = std over all G rewards + 1e-4

**Per-token advantages** (shape B × T, which TRL's loss accepts):
- on-policy completion i, every policy token: (r_i − V̂)/σ;
- forced branch of class a, the forced decisive-action tokens: (Q̂(a) − V̂)/σ;
- forced branch, the sampled continuation tokens: 0. They are on-policy given the forced turn, but a single continuation has no same-state baseline.

**Off-policy weight.** The forced tokens are logged with sampling log-probability 0, i.e. μ = 1 for a deterministic forced action. TRL's sequence-level importance correction then gives the forced branch:

  ρ = π_old(forced tokens) · exp(Σ_continuation (log π_old − log μ)) ≈ π_old(a | h*).

Since ρ ≤ 1, the `clip_max = 3` mask never removes a forced branch.

**Gradient at the decision state.** At c = 1 (first iteration), the gradient from forced branch a is ρ·(Q̂(a) − V̂)/σ · ∇ log π(a) = (Q̂(a) − V̂)/σ · ∇π(a|h*). Summed over classes, this is the **all-action (expected) policy gradient** Σ_a ∇π(a|h*)(Q(a) − V) at h*, estimated with environment-forked Monte Carlo Q values. The on-policy rollouts add the usual likelihood-ratio estimate of the same gradient, plus the gradient at later turns.

## 3. Why this targets ActionProof's failure

At decision states where the correct recovery is rare under π_old, groups of on-policy rollouts rarely contain it: the zero-variance problem of §2 in the Stage 3 report. Recovery starts made such states frequent; FCA additionally makes every relevant action *evaluated* at them, independent of whether the policy samples it.

Two consequences:
- An action the policy never takes still receives a gradient proportional to π(a)(Q(a) − V), which is small but correctly signed.
- A frequent wrong action, such as proceeding after a partial payment, gets a negative signal from its own forced branch even when all on-policy rollouts fail identically. In that case standard GRPO's group has zero variance, but σ here includes the forced branches.

## 4. Information isolation

- Forced actions are functions of h* only (§1).
- The verifier's state enters only through r.
- The `fork_meta` that the rollout passes to the trainer holds group id, branch kind, action class and token span. It never reaches a prompt.
- Tested in `tests/stage4/test_fca.py`: forced-action arguments are recomputed from a conversation with every hidden field removed; the prompt text of all branches is identical up to the decision state.

## 5. Cost

- Same number of rollouts per step as Stage 3 (4 prompts × 8).
- About half the groups are decision-state groups, so about 16 of 32 rollouts per step are forced branches.
- Generated tokens per forced branch are lower by the length of one turn.
- No extra model forward passes: the importance weight comes from TRL's existing old-log-probability pass.
- Simulator overhead: K extra prefix replays per group (milliseconds).

## 6. Assumptions and failure modes

- **Q̂ is high-variance.** It averages 1–3 samples per class.
- **The class partition is hand-defined per domain.** Classes outside K(h*) get no forced evaluation.
- **Credit to the forced turn only.** Continuation tokens of forced branches are not trained, so FCA gives credit to the decision itself, not to the recovery that follows.
- **Loss of on-policy samples.** Forced branches cost 4 on-policy samples per decision-state group. If Q̂ adds little information, FCA can be worse than recovery-start GRPO.
- **Inconsistent estimators.** The two estimators of the decision-state gradient are summed with weight 1 each. This is a scaled, not an exactly unbiased, combination of two unbiased estimators.
- **Mismatch with the old policy.** `sequence_mask` uses π_old from the trainer's pass, so a tokenisation mismatch between forced text and model tokens would make ρ wrong. The test re-tokenises the forced turn exactly as the rollout does.

## 7. Falsification (kill criteria, fixed before training)

Development data only, against recovery-start GRPO (the same pool, start states, steps, rollouts and seeds):
- FCA is dropped if `s3_rec_dev` VSC is not at least +0.03 above recovery-start GRPO, averaged over development seeds 0–1;
- or if `proc_dev` VSC falls by more than 0.03;
- or if training diverges (KL > 10× recovery-start GRPO's at step 100).

A confirmatory comparison is registered only if FCA survives these criteria.
