# Final story decision

2026-11-27 (Phase B of the Stage 2 sprint). Inputs:
- `research_audit/CURRENT_BRANCH_FORENSIC_REVIEW.md`
- `research_audit/NOVELTY_POSITIONING_FINAL.md`
- `research_audit/COMPETITIVE_THREAT_RANKING.md`

**Placeholders.** Sections that depend on SEL1 say so. They are filled only from `results/sel1/` after its
registered analysis runs.

## The five candidate stories and the choice

| Story | Role |
|---|---|
| C: transaction-integrity benchmark (identical replies hide different database states; acknowledgement trust and retries fail) | **Primary spine**, together with B |
| B: recovery learning (demonstrations carry the capability; GRPO adds an exploratory increment after SFT) | **Primary spine**: the trained agent that the benchmark measures |
| D: selective recovery and safe escalation | **Supporting finding 1.** It is the open problem the benchmark exposes (H5), and SEL1 tests a fix |
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

## 2. Central claim (one sentence; SEL1-dependent part marked)

On a locked benchmark where identical "ok" and "timeout" replies hide committed and uncommitted payments:
- recovery demonstrations followed by GRPO raise a 3.35B Cohere agent from 0.513 to 0.819 verified completion on
  450 frozen tasks;
- the agent duplicates mutations far less often than bounded retry with idempotency keys (0.024 against 0.191);
- but it rarely hands off when evidence cannot settle the outcome, which `[SEL1: fixed / not fixed]`;
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
3. **(Conditional) selective escalation (SEL1).** `[filled from results/sel1/ after the registered analysis]`.
   If SEL1 fails, contribution 3 becomes "the escalation gap, measured", reported with always-escalate and
   never-escalate references.

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
- The scripted reference B4 still beats every learned agent on TXN (VTC 0.864 against 0.724).

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
