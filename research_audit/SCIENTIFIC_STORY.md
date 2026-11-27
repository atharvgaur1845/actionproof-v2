> **Superseded (2026-11-27, Stage 2 sprint).** This file is kept as history (last updated 2026-11-25) and is not current. The current source is `docs/submission/FINAL_STORY_DECISION.md`. See `research_audit/STALE_SURFACES_AUDIT.md`.

# Scientific story

2026-11-25. Every claim names its ledger ID (`claims/claims.yaml`) and evidence level:
- **[C]** registered and confirmatory (rule met);
- **[U]** registered, rule not met;
- **[S]** registered secondary;
- **[X]** exploratory or descriptive;
- **[D]** development;
- **[F]** failed by its own criterion.

Recomputation status is in `CLAIM_VERIFICATION.md`.

## 1. Questions

**Primary.** When a write tool's success reply cannot prove the write happened, what does reinforcement learning teach a tool-using language agent? Specifically, does the information source of the reward decide whether the agent verifies its own writes?

**Secondary.**
1. Does verification lead to correct recovery?
2. Which interventions teach recovery: curriculum, demonstrations, inference-time evidence, runtime guards?
3. Why does group-normalised RL learn verification the way it does?
4. Can LLM judges replace the deterministic verifier?
5. Do the learned behaviours transfer, and do they cost general capability?

## 2. Five competencies, and what the evidence says about each

| Competency | Strongest evidence | Verdict |
|---|---|---|
| **1. Evidence acquisition** (reading after a write) | A12 [C]: database reward gives read-back 1.00 vs 0.00 under the reply reward, 5 of 5 seeds (recomputed exactly). ID1 [C]: from a warm start that never reads, the database reward never samples a read in 200 steps | RL **selects** reading from the warm start's repertoire; it does not create it |
| **2. Evidence interpretation** | A14 [C]: with the read forced into context, every model, even one that never reads, distinguishes present from absent payments. A15/A16 [C]: all models linearly encode partial vs complete | Interpretation is **present** before RL. A16's probes show decodability only; its patching moved the action margin, but the ablation effect was not specific to R_V |
| **3. Next-action selection** | A14 [C]: on wrong-amount payments only the database-reward agent refunds, in 0.35–0.48 of tasks; duplicates are left in place by both. E2 [C]: a reconciliation line raises correct refunds after a goods receipt by +0.48 | Selection is where agents fail. Checking is not recovering |
| **4. Recovery and completion** | S3 [C]: recovery-start curriculum +0.21 over the state-reward agent (recomputed +0.206/+0.209). P1 [C]: recovery SFT +0.259 and recovery SFT + same GRPO +0.306 over the recovery-start agent, on 450 fresh tasks (recomputed). AD-C1 [C]: no-std GRPO +0.120 over GRPO in every seed | Recovery is **learnable from demonstrations**. Inference-time text (P1 B, P2 B2) and a naive guard (P2) hurt |
| **5. Honest reporting** | P1 descriptive [X]: false success falls from 0.385 (A) to 0.110 (D). The P2 guard removes false success almost entirely, but adds missing reports and under-claims | Honesty improves as a by-product of recovery training. Forcing it at run time costs completion |

## 3. Causal structure (what was intervened on, with controls)

- **Reward information source.** A12 holds warm start, tasks and reward form fixed, and varies only the channel. A **causal intervention** with paired seeds [C].
  - Its composite registered claim C1 failed, because H3 (the per-class mechanism prediction) failed [U].
  - K1 [C] shows that training-pool composition explains part of R_A's drift, not its zero read-back.
- **Start state.** S3 addendum 2 compares recovery starts with a class-matched control, five runs per arm: +0.107 [C]. This is a causal intervention on start states.
  - The partial-payment-specific effect was not supported [U].
  - S3's +0.21 over A12 R_V is a **recipe** comparison: it also adds fault classes, and the duplicate-delivery gain is explained by coverage (S3-A1).
- **Demonstrations.** P1 C and D vs A [C] are interventions on training data. D − C was not registered, so the extra GRPO stage is not established.
- **Normalisation.** AD-C1 [C] intervenes on the advantage normaliser only, holding everything else fixed: no-std beats GRPO, in line with the theorem, whose seven testable predictions T1–T7 all held. This confirms an existing method's advantage (Dr. GRPO) in this setting.
- **Correlational only:**
  - A16 probes (decodability);
  - the mechanism diagnostics on training logs (development);
  - J1/J2 judge behaviour, which is descriptive against the verifier.

## 4. Hypotheses that were falsified, and what each ruled out

| Falsified | What it ruled out | What remains open |
|---|---|---|
| A12 H3, the per-class "punishment" mechanism [U] | That the reply reward suppresses reading through the specific per-class advantage pattern predicted | Why R_A ends exactly at the warm start |
| A14's original thesis (abandoned) | That trained agents differ in how they react to evidence: they react the same in the same situation | How often agents reach the situation where they would read |
| Partial-specific start-state effect [U] | That recovery starts help partial payments specifically | Whether handed evidence matters (addendum 4 inconclusive) |
| Timeout-before-commit stress (Stage 1) | That PCSTR-d is robust to pre-commit timeouts (VSC 0.176) | Robustness to that fault in Stage 2 |
| Transfer (A17, S3-R, S4) [U] | That learned verification transfers to new domains, tool names or workflow orders | Generalisation beyond the trained procedure |
| SR1 tool-name randomisation [U] | The standard fix for surface overfitting, in this setting | — |
| FCA, C5, G, CDVO, D2 [F] | Five credit-assignment and sampling candidates | — |
| P1 H1, P2 H1/H3/H4, S1–S3 [U] | Inference-time hints, ledgers and a naive runtime guard as substitutes for training | A guard that permits the taught repair (not tested) |
| TH1 H6 [U] | That GRPO is never worse than Dr. GRPO when the vote is right (it is slow under large stakes) | — |
| H_AWRB (AD-C1) [U] | That AWRB adds anything over a plain centred baseline | — |
| CircuitKIT EAP-IG (exploratory) | Nothing: it never ran. It failed importing a transformers class (D-A16-1). No circuit claim is made; only probes and patching with controls are used | Circuit-level explanation |

## 5. Alternative explanations to keep in view

- **Warm-start dependence.** A12, ID1 and AD-C1 all show RL amplifying the warm start's behaviour. Results could differ from another warm start.
- **Single base model** (Tiny Aya Global 3.35B) and a **single simulator family**. ProcureSim's faults are stylised.
- **VSC ignores the cost of reads.** AWRB's +0.192 comes with heavy over-verification, about 2.7 reads per episode.
- **Development choices.** Several method choices (curriculum mixture, recovery SFT data) were made on development splits. The test splits were never used for choice.
- **The scripted oracle** (0.924) still beats the best trained agent (0.819) on the same tasks. It knows the fault taxonomy.

## 6. Claims that exceed the evidence (remove or weaken)

- "GRPO learns verify iff p·s > 1/2 irrespective of cost": true only for state-level repairability (Phase II audit).
- "Dr. GRPO's update is proportional to the value of information": it is proportional to the on-policy action-value gap ΔQ.
- Any claim of a **new algorithm**. AWRB and D3 are development results, and AWRB's own contribution failed AD-C1.
- "Read-back after every phantom write": say "after every phantom write that reached the database". Idempotency-store replies are excluded.
- Anything implying that LLM judges can certify outcomes.
- Any causal claim from A16 beyond "the representation exists and is decodable".

## 7. The smallest coherent paper

**One problem, one mechanism, one confirmed fix, one practical result:**
1. **Problem:** silent write faults, measured with a deterministic verifier that grades on database state, never the transcript.
2. **The reward channel selects verification** (A12, ID1).
3. **Mechanism:** with a shared hidden state, group-normalised updates are stake-blind (Phase II theorem, TH1), confirmed in the trained agent (AD-C1).
4. **Practical result:** recovery is taught by demonstrations, while inference-time evidence and naive guards fail (P1, P2).
5. **What does not transfer:** new workflows, renamed tools, other domains.

## 8. Revised abstract

Tool-using agents treat a write tool's "ok" as proof that a payment or record was stored. We built ProcureSim, a procurement simulator whose success replies are byte-identical whether a write stored everything, part of it, two copies or nothing, and graded agents with a deterministic verifier that reads the database.

Training Tiny Aya Global (3.35B) with GRPO from one warm start, a reward that reads the database made the agent read back after phantom writes in five of five seeds (1.00 vs 0.00). A reward of identical form that reads only the reply did not. A warm start that never reads stayed at zero reads, so the reward selects verification rather than creating it.

We show why. When every rollout in a group shares one hidden fault, GRPO's group normalisation discards the stakes and costs of verifying. Its expected update becomes a stake-free vote, while mean-centred estimators follow the value gap. A registered held-out simulation predicts GRPO's learned decision in 721 of 721 problems. A registered test in the trained agent confirms the predicted fix: removing the normaliser raises verified completion by +0.120 in every seed.

Checking is not recovering. Recovery demonstrations followed by the same RL raise verified completion on 450 fresh tasks from 0.513 to 0.819. Inference-time hints, a structured ledger and a deterministic runtime guard each lower it.

Gains hold under Hindi and Hinglish, but not under renamed tools or a new workflow order, where every agent skips the new step and reports success. LLM judges told that "ok" is not proof abstain rather than certify outcomes. With a database snapshot they agree with the verifier, but none is verifier-grade.

Every experiment was registered and hashed before it ran. Negative results are reported with what they rule out.

## 9. Contribution statement

ActionProof contributes three things. First, a controlled environment and verifier that separate what a tool reply says from what the database holds. Second, registered evidence that the reward's information channel selects whether an agent verifies. Third, an exact account, confirmed in a registered held-out simulation and a registered language-agent test, of why group-normalised RL is blind to the stakes and costs of verification when rollouts share a hidden state. The practical consequences are that recovery must be taught with demonstrations, and that verification learned this way does not transfer to unseen workflows. No new optimiser is claimed. The confirmed fix, removing the group standard deviation, is Dr. GRPO's.
