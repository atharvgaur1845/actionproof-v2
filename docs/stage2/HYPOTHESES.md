# Stage 2 hypotheses (from PLAN.md §D, revision 3)

## Research framing

**The research and the product are framed separately.**

**Research:** *Reward-Information Observability and Evidence Acquisition in Tool-Using RL.*
- The novelty is a controlled, training-time manipulation of what the reward can see, under phantom writes that look identical to real ones.
- It is replicated in a second, separately implemented procurement environment with five independent simulated services.
- The mechanism is predicted from a reward-truth table before training.
- We also measure where learned evidence acquisition fails to turn into recovery.

**Product:** *ActionProof: Verifiable Execution Infrastructure for Agentic Workflows.*
- An **execution verification and recovery layer**, not a distributed transaction manager. It gives no atomicity across services.
- Components: contracts, authoritative-state verification, bounded recovery, and evidence provenance.

**Why the three-layer question is not the headline.** LIMBO ("Where Does Exactly-Once Live?") already varies model, harness and tool contract over 25,930 episodes. 2611.23353 and Cordon already cover the runtime. The policy / contract / runtime split stays only as the frame for evaluating the product.

**Claims** (each is reported however it comes out):
- **C1, novelty (P0): reward observability.**
  - On writes that return a success acknowledgement without committing, a verdict-grounded reward selects authoritative read-back, and an acknowledgement-grounded reward of identical form does not.
  - The sign is the same in every paired seed (minimum 3, target 5).
  - The per-class mechanism is predicted before training.
  - Measured too: where acquisition fails to integrate into recovery ("read, then ignore").
- **C2, measurement (P1): contract coverage.** Which violations a contract catches; which it misses when one clause is removed; which it misses through clause interactions; and which it cannot express at all. Tested on faults generated independently of the contract.
- **C3, measurement (P0): policy–runtime contributions.** We *estimate*, without presupposing a direction, three quantities per gateway mode and fault cell:
  - Δ_policy, the learned policy's contribution;
  - Δ_runtime, the deterministic runtime's contribution;
  - Δ_information, the extra value of exposing evidence to the agent.

| Layer | Question | Primary (P0/P1) | Supporting |
|---|---|---|---|
| Policy | Does a verdict-grounded reward select evidence acquisition where an acknowledgement-grounded reward cannot? | **A12** (H2; seeds ≥ 3, target 5), **H3** | H8 recovery choice; H16 (P3) |
| Contract | What can a contract establish, miss, or not express? | **H5** coverage generalization (A13) | H6 (P3); H9 (P2); H14 (consistency check only) |
| Runtime | What does enforcement add, and what does exposing evidence add? | **H1** estimands (A11); **H7** | H10; H12 (P2, deferred if P0 slips); H11 |
| Cross-cutting | Does execution evidence improve evaluator decisions? | **H4** audit (Track 3 item) | — |

**Hypotheses** (revision 3 changes: H2's primary endpoint, H3 with a reward-truth table, H1 as estimands, H5 as coverage generalization, H4 between-subjects, H14 demoted):

| # | Hypothesis | Primary metric | Falsified if | Role |
|---|---|---|---|---|
| H2 | **Case A (primary): writes whose first acknowledgement is success but which did not commit.** On untouched `proc_conf`, **phantom read-back** is ≥ 15 pp higher under R_V than R_A, with the same sign in every paired seed. Secondary: verified completion gap ≥ 10 pp. Check: within each arm, read-back after benign success acknowledgements equals read-back after phantom ones (observation equivalence). **Case B (separate diagnostic, error acknowledgements):** lost-ack read-back, precommit recovery and false success, reported per arm with no directional claim | phantom read-back | Hierarchical CI includes 0, or the sign differs in any seed | **P0** |
| H3 | Before step 1 we commit (i) the **reward-truth table** (§G3) and (ii) per-class predictions for the sign of verification's relative advantage under each reward, from step-0 gate samples. Planning-stage expectation (superseded): under R_A, negative in lost-ack and never-commit classes, slightly negative on phantom; under R_V, positive on phantom and timeout before commit. **The registered predictions are those of `PREREGISTRATION_A12.md` §3 and §7 and `results/a12b/h3_predictions.json`; where they differ from this sentence (for example R_V on timeout before commit: 0 in the registered truth table, negative in the a12b step-0 file), the registration governs.** Training logs (per-class group advantages, informative-group counts) and per-class read-back changes are then compared with the predictions | sign agreement per class | Registered criterion (A12 §7): signs agree in ≥ 80% of classes with a non-zero prediction | **P0** |
| H1 | Estimation over 4 modes × 7 policies × fault cells on a frozen multi-axis taxonomy (§G1). Δ_policy(m,c) = VSC(π_trained,m,c) − VSC(π_warm,m,c); Δ_runtime(π,c) = VSC(π,Enforce,c) − VSC(π,Observe,c); Δ_information(π,c) = VSC(π,Enforce+Inform,c) − VSC(π,Enforce,c). Registered two-sided test: does the runtime change the policy's contribution? That is, Δ_policy(Enforce,c) − Δ_policy(Passthrough,c) ≠ 0, pooled over full-observability cells. Engineering check: Δ_runtime > 0 on full-observability, safe-recovery cells | the three estimands, with CIs, per cell | No directional prediction. If the engineering check fails, the runtime does not work as designed and is reported as such | **P0** |
| H5 | **Coverage generalization**, on three fault groups: G-rep (what the contract represents), G-comb (combinations of known primitives), and G-adv (adversarial faults written by an independent agent from transaction semantics alone, without seeing the contract). Measured: clause necessity (single removal); redundancy; pairwise interactions (pair leak > sum of single leaks); composition generalization (G-comb); specification gap (G-adv violations no clause can express). Prediction: full-contract false acceptance on G-adv > on G-rep (one-sided) | false acceptance per group; interaction table | False acceptance on G-adv ≤ on G-rep | **P1** |
| H7 | Idempotency keys alone recover 0% of phantom and partial writes. No verification-only policy is exactly-once under unbounded late commits (LIMBO Prop. 1). Keys plus reconciliation leave at most one irreversible effect under stated provider assumptions. Same fault traces for every defence | recovery; duplicates; residual exposure | Keys recover ≥ 30% of phantoms | **P1** |
| H4 | Single-annotator **audit**, between-subjects to avoid carryover. 60 stratified episodes (20 invisible failures, 20 visible failures, 20 true completions), randomly split 10/10 within each stratum into transcript-only and ledger-visible, each labelled once. Prediction: transcript-only false acceptance on invisible failures ≥ 0.5; ledger-visible ≤ 0.2. Judge (TaskCompletion; ledger-instructed 32B) vs human per condition, with confusion matrices, raw agreement and κ with CIs | false acceptance per condition × stratum; κ | Transcript-only false acceptance on invisible failures < 0.3 | **P1** |
| H12 | VOI(a) = E[L \| no verification] − E[L \| a] − C(a). Six policies: never, always, fixed rate, random at equal cost, amount threshold, expected risk. Pareto frontier; raw exposure reported separately; sensitivity over loss weights | frontier | Expected-risk is no better than random at equal cost | P2 (deferred if P0 slips) |
| H9 | Local verification gives ≥ 90% correct writes but ≤ 70% cross-service invariants under partial or cross-service faults; cross-service clauses restore ≥ 95% | invariant satisfaction | Local already ≥ 95% global | P2 |
| H8 | Trained agents' recovery ignores what their read showed; a deterministic engine reaches ≥ 0.9 recovery on resolvable cases and ≥ 0.9 escalation recall on unresolvable ones | recovery; recall | Engine < 0.9 | supporting |
| H10 | Stale replicas and forged receipts in non-authoritative output are caught by version freshness and corroboration. A lying authoritative source without corroboration gives AMBIGUOUS and escalation, which is reported as a limit | false finalization | Finalizes success on forged evidence | supporting |
| H11 | Exposure-weighted loss changes at least one policy ranking (with weight sensitivity) | expected loss; raw exposure | Rankings identical | supporting |
| H14 | Remaining false commitment under report gating vs the H5 leak union | — | — | **engineering consistency check, not a claim** (it may follow algebraically from the shared scorer) |
| H6, H16 | Aya-drafted contracts; transfer to a held-out service | — | — | P3 |
| H13, H15, H17 | — | — | — | deferred |
