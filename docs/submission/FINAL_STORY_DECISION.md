# Final story decision

2026-11-27 (Phase B of the Stage 2 sprint). Inputs:
- `research_audit/CURRENT_BRANCH_FORENSIC_REVIEW.md`
- `research_audit/NOVELTY_POSITIONING_FINAL.md`
- `research_audit/COMPETITIVE_THREAT_RANKING.md`

**Revision after the final hostile review (2026-11-27, committed 10:39 IST).** Three independent reviews (Reviewer 1: ML
research; 2: systems; 3: hackathon judge; `research_audit/FINAL_REDTEAM_REVIEW.md`) found construct problems in the
TXN benchmark and in what SEL1's tests measure. They are recorded in `PREREGISTRATION_TXN1_ADDENDUM3.md` and
`PREREGISTRATION_SEL1_ADDENDUM1.md`. The spine does not change. What changes:
- SEL1 is described as "taught hand-off after repeated outage errors", not "fixed selective hand-off".
- The benchmark's limits are stated next to its results.
- A false statement that both SEL1 arms pass B4's VTC is corrected (only CTL does; SEL's 0.830 is below 0.864).

**SEL1 filled (2026-11-27, 09:57 IST).** The SEL1-dependent parts below were filled from
`results/sel1/sel1_summary.json` after task 1812 ran the registered analysis. SEL1 is supported (H1, H3, H4); H2
passes; transfer to a held-out step is seed-fragile.

## The five candidate stories and the choice

| Story | Role |
|---|---|
| C: transaction-integrity benchmark (identical replies hide different database states; acknowledgement trust and retries fail) | **Primary spine**, together with B |
| B: recovery learning (demonstrations carry the capability; GRPO adds an exploratory increment after SFT) | **Primary spine**: the trained agent that the benchmark measures |
| D: selective recovery and safe escalation | **Supporting finding 1.** It is the open problem the benchmark exposes (H5); registered SEL1 (supported) taught hand-off after repeated outage errors; where hand-off is the only safe action it happens in 164 of 432 episodes, seed-fragile (SEL1 addendum 1), with seed-fragile transfer to a held-out step |
| A: the reward's information source selects read-back but does not create it (A12, ID1) | **Supporting finding 2.** One line in the main text; detail in the appendix and website |
| E: group-normalisation theory | **Companion theory paper** (`paper/main_v3.tex`). It is not part of the main narrative |

**Why E is separate.**
- The theory is about costly verification under a shared hidden state. In ProcureSim, reads are free.
- The registered test of its fix on the practical agent (P3) found no gain (−0.007).
- Forcing it into the main story would make the claim incoherent.

It stays a labelled companion artefact, with its own registered evidence (TH1, AD-C1). The landing page says:
"Companion theory paper (separate claim): `paper/main_v3.pdf`."

## 1. Primary research question

When a write tool's acknowledgement does not reliably describe what happened, can a small open model be trained
to reconcile the evidence? That means:
- verify before declaring success;
- repair partial and missing writes without duplicating them;
- hand off when the outcome cannot be established;
- and do this measurably better than the scripted baselines engineers use today.

## 2. Central claim (one sentence)

On a locked benchmark where identical "ok" and "timeout" replies hide committed and uncommitted payments:
- recovery demonstrations followed by GRPO raise a 3.35B Cohere agent from 0.513 to 0.819 verified completion on
  450 frozen tasks;
- the agent falsely completes far less often than one that trusts the reply (0.121 against 0.864), and leaves fewer
  duplicates than bounded retry with idempotency keys, both those it could cause and those the environment injects;
- a hand-written reconciliation script still beats it, and the benchmark has construct limits found by our own
  review (TXN1 addendum 3);
- registered SEL1 taught hand-off after repeated outage errors at no cost on resolvable tasks; where hand-off is the
  only safe action it happens in 164 of 432 episodes, seed-fragile;
- every result comes from a procurement simulator.

## 3. Contributions

1. **The TXN benchmark and the reference-policy ladder.**
   - 440 locked tasks: counterfactual quadruples, eight fault classes, three evidence regimes, six OOD groups.
   - Scripted policies B0–B4 and ablations A1–A6 under one evaluator.
   - 25 of 25 mutants killed.
   - Released episodes and splits.
   - The paired design is shared with "Did It Happen?"; the composition and the released data are ours.
2. **Trained recovery under ambiguous execution.**
   - P1 registered, three seeds: A 0.513 → C 0.772 → D 0.819.
   - P4: randomised demonstrations remove procedure replay **in the warm start**.
   - P5: the recipe on four other Cohere bases, one SFT seed each.
   - R7 (exploratory): GRPO adds +0.140 on top of its own warm start.
3. **Selective escalation (SEL1, registered before training; supported).** Demonstrations from a scripted
   selective rule (B5), against a matched-volume control from the same Dw_aug parents, 3 seeds:
   - H1: escalation on the locked test's unresolvable tasks 0.972 against 0.002 (+0.970 [+0.952, +0.987]);
   - H3: selectivity J 0.740 against 0.007 (+0.732 [+0.625, +0.836]); 0 of 360 escalations on a transient outage of
     held-out length;
   - H4: no VTC cost on resolvable tasks (+0.002 [−0.006, +0.010]);
   - H2 (held-out compositions) passes, but on the held-out step one seed escalates in 2 of 120 episodes.
   The gap is measured with always-escalate and never-escalate references beside it (J 0 for both).
   - Post-hoc scope (SEL1 addendum 1). Every H1 task could be settled by a same-key resend. The audit-log-only
     control cannot fire, and the 2-read outage is inside the demonstrated budget. Where hand-off is the only safe
     action, 164 of 432 episodes.

## 4. Evidence file for every contribution

| Contribution | Registration | Result file | Analysis |
|---|---|---|---|
| 1 | `PREREGISTRATION_TXN1.md` + addenda 1–2 | `results/txn/stats__txn_test.json` | `scripts/txn_stats.py` |
| 2 (P1) | `PREREGISTRATION_P1.md` | `results/p1/p1_summary.json` | `scripts/p1_stats.py` |
| 2 (P4) | `PREREGISTRATION_P4.md` | `results/p4/p4_summary.json` | `scripts/p4_stats.py` |
| 2 (P5) | `PREREGISTRATION_P5.md` + addendum 1 | `results/p5/p5_summary.json` | `scripts/p5_stats.py` |
| 2 (R7) | exploratory | `results/audit/r7_grpo_increment.json` | `scripts/audit/r7_*` |
| 3 | `PREREGISTRATION_SEL1.md` | `results/sel1/sel1_summary.json` | `scripts/sel1_stats.py` |

## 5. Negative results that constrain the claim

- TXN H3 is not supported as a family: there are fewer duplicates, but not fewer unnecessary retries.
- H5: D escalates in 15 of 540 cases; Dw_aug in 0 of 540, with 53 of 540 unsafe.
- D fails renamed tools (0.20) and the four-step workflow; P4's fix is at the warm-start stage only.
- P3 (no-std GRPO) is not supported.
- SR1 (renaming tools during GRPO) failed.
- Healthcare transfer is negative.
- LLM judges are not verifier-grade.
- The scripted reference B4 still beats every learned agent on TXN wherever the evidence is observable (VTC 0.864
  against D's 0.724). SEL1's SEL arm (0.830) is below it. Its CTL arm (0.871) passes its overall VTC only on the
  quadruples, by settling unresolvable tasks with same-key resends where B4 hands off (descriptive; corrected after the
  hostile review, which caught an earlier "both arms pass").

## 6. What the project does not claim

- No new RL optimizer.
- No novel benchmark *design*.
- No claim that the agent is safe or generalises beyond the stated splits.
- No claim of real-system validation.
- The decision record is a diagnostic record, not a proof.
- The reference verifier is a consistency check (same author and specification), not independent ground truth.
- External models are supplementary context only.

## 7. How the pieces fit

**Benchmark (TXN)** → it measures whether an agent tells committed, missing, partial and ambiguous outcomes apart.
↓
**Trained agent (D, Dw_aug, SEL1 arms)** → measured on it, against scripted and prompted baselines.
↓
**Live Space** → one curated task and reality at a time. A real Cohere model runs; the database is revealed
afterwards. This is an *illustration*, not the benchmark.
↓
**Replay site and traces dataset** → the full registered study, every episode inspectable.

The theory paper is a separate, labelled companion.

## 8. What goes where

| Surface | Content |
|---|---|
| Six core slides / video | The counterfactual "ok" (TXN) → why naive fixes fail (B0/B1/B3 against B4) → system and benchmark → what training changes (P1, P4; P5 as breadth) → what still fails (H5 / SEL1, H3, D's transfer) → impact and reproduction |
| README (landing page) | The thesis, a numbered 60-second judge path, four headline tables (P1, TXN, P4/P5, SEL1), limits, artefacts and reproduction. History moves to `docs/HISTORY.md` |
| Write-up main text | As the README, plus methods, metrics, statistics, negative results and the declarations |
| Appendix (write-up §§ and website) | A12/ID1 (reward information), S3/S4, E1/E2, J1 judges, R1 capability, TH1/AD-C1, A15/A16 |
| Website (replay Space) | Every registered result with files; the episode explorer |
| Live Space | Judge mode: one guided scenario, base against trained against scripted, state reveal, decision record, export |
| Companion paper | `paper/main_v3.tex`: the group-normalisation theory, TH1, AD-C1, P3 |
