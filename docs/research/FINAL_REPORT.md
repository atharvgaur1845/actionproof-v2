# ActionProof Stage 4: final report

Branch `stage4`. Every number below resolves through `claims/claims.yaml` (`python scripts/s2_numcheck.py`) or names its result file. Sections marked **PENDING** wait on registered runs that are still training. They are filled only from the named files.

## 1. What changed

- **Research.**
  - A novelty audit with 84 verified entries.
  - FCA (forked counterfactual credit), implemented, tested and **dropped** by its pre-set criterion.
  - S4-EVAL: language, API-surface, workflow and fault-combination shifts, registered and run.
  - Two new registrations: S3 addendum 2 (five control seeds; workflow-rebalanced pools) and ID1 (does the state reward create read-back, or select it?).
  - A post-results academic review, with all of its wording corrections applied.
- **Paper** (`paper/main.tex`, 5 main pages plus a 4-page appendix): reframed around the setting and registered measurements. It claims no new optimiser, and its claims were narrowed to the evidence.
- **Product** (`product/`, local only, not deployed):
  - live: incident simulator, execution-truth timeline, reward lab, recovery arena;
  - recorded: a policy comparison;
  - generated: a research dashboard, now with S4 and FCA.
- **Integrity tooling.**
  - Number checking: 241 numbers, 0 failures. The checker no longer needs an allow-list entry that could hide a result.
  - Three hash-frozen registrations this session (S4-EVAL addendum 1, S3 addendum 2, ID1).
  - Four logged deviations (D-S4E-1, D-S4E-2, D-ID1-1 and the H4 rounding correction).

## 2. The new algorithm

There is none. Every candidate (A: forked counterfactual credit; B: recovery-aware constrained GRPO; C: adaptive recovery frontier) is published, and so is the Stage 3 recovery-start curriculum (`docs/research/NOVELTY_AUDIT.md`, `docs/research/algorithm/DECISION.md`). FCA was carried forward as an honest test of a known estimator family in this setting, not as a contribution.

## 3. What we tested

| Experiment | Registration | Status |
|---|---|---|
| S3: recovery-start recipe (RSC) vs A12 R_V, 5 seeds | `PREREGISTRATION_S3.md` | done (earlier) |
| S3 addendum 1: RSC vs class-matched control, 3 seeds | `PREREGISTRATION_S3_ADDENDUM1.md` | done (earlier) |
| FCA development test, 2 seeds | `docs/research/algorithm/FCA_GRPO.md` §7 | **done: dropped** |
| S4-EVAL: 7 shift splits, 14 frozen models | `PREREGISTRATION_S4_EVAL.md` + addendum 1 | **done** |
| S3 addendum 2 Part A: control seeds 3–4 | `PREREGISTRATION_S3_ADDENDUM2.md` | PENDING |
| S3 addendum 2 Part B: workflow-rebalanced pools | same | PENDING |
| ID1: no-verification warm start, then R_V | `PREREGISTRATION_ID1.md` | PENDING |

## 4. Results

**S4-EVAL** (`results/s4/s4_summary.json`). RSC minus A12 R_V, VSC:

| Hypothesis | Shift | Difference [95% CI] | Seeds positive | Holm p | Verdict |
|---|---|---|---|---|---|
| G1 | renamed API | −0.016 [−0.044, 0.011] | 3/5 | 0.82 | not supported |
| G2 | Hindi | +0.127 [0.080, 0.176] | 5/5 | 0.016 | supported |
| G2 | Hinglish | +0.072 [0.040, 0.109] | 5/5 | 0.016 | supported |
| G3 | unseen four-step workflow | +0.016 [0.004, 0.030] | 2/5 | 0.82 | not supported |
| G4 | two faults per episode | +0.072 [0.032, 0.118] | 5/5 | 0.016 | supported |

Diagnostics (`results/s4/s4_diagnostics.json`, descriptive):
- **Renamed API.** Trained agents still call training-time tool names in 13–35% of episodes (warm start 62%).
- **Four-step workflow.** Every model replays the trained three-step sequence and leaves the goods receipt undone in 84–96% of episodes. Each trained arm's mean VSC is below the warm start's (individual runs vary: one control run scores 0.27).
- **Language cost.** Hinglish costs RSC 0.09 VSC against English on the same tasks, twice A12 R_V's cost.

**Seed-level and per-class S3** (`results/s3/s3_seedlevel.json`, `results/s3/s3_review_numbers.json`):
- Every RSC run beats every A12 R_V run (p = 0.004).
- Of the full-episode gain, 0.12 is duplicate delivery (a class A12 never trained on) and 0.09 is partial payments.
- Against the control, partial repair was learned in 5 of 5 RSC runs and 1 of 3 control runs. The control is equal or better on duplicates.
- Partial payments after a goods receipt stay unrepaired by every arm (RSC 5 of 60).
- Transfer to healthcare administration is negative: 0.158 → 0.080 → 0.047.

**PENDING:** S3 addendum 2 (`results/s3/s3_addendum2.json`) and ID1 (`results/id1/id1_summary.json`).

## 5. What failed

- **FCA** (`docs/research/algorithm/FCA_RESULT.md`).
  - −0.097 development VSC against recovery-start GRPO; at least +0.03 was required.
  - Seed 0 did not learn partial repair.
  - Its forced branches did carry signal in 66–72% of fork groups, but recovery did not improve.
- **Robustness to the API surface and to a new workflow step** (G1, G3).
- **Transfer to sibling domains** (negative in healthcare administration).
- **Process slips, all logged:**
  - an unattainable Holm rule (fixed before outputs);
  - a statistics-script pairing bug (fixed by wrapper, D-S4E-2);
  - a missing evaluation flag (D-ID1-1);
  - three timestamp slips in the decision log, corrected the same session.

## 6. Novelty verdict

- **Not novel:** any optimiser or credit estimator here; the recovery-start curriculum; the observation-equivalent paired evaluation design. The concurrent "Did It Happen?" preprint uses the same design and is cited by URL.
- **Defensible:**
  - the setting: RL training with silent write commits and duplicate risk, unmatched among the papers checked;
  - the registered measurements in it: observability selects read-back; acquisition dissociates from use; repair is bound to workflow and tool surface;
  - a per-class decomposition of a known recipe;
  - the process: hashed registrations, dated deviations, a number-checked ledger.

## 7. Product verdict

Local Gradio app (`python -m product.app`). Its 6 tests pass, and it has been checked in the browser on desktop and mobile widths. Each part is labelled live, recorded or generated. It is not deployed and makes no model calls.

## 8. Consistency fixes

`docs/CONSISTENCY_AUDIT.md`: items 1–18 (round 1) and 19–35 (round 2, from the post-results review). No registration was edited. Every correction is dated.

## 9. Reproduction

- **Numbers:** `python scripts/s2_numcheck.py`.
- **Statistics:** `scripts/s3_stats.py`, `scripts/s4_stats_d2.py`, `scripts/s3_stats_a2.py`, `scripts/id1_stats.py`, `scripts/s4_fca_dev.py`, `scripts/s4_diagnose.py`, `scripts/s3_review_numbers.py`.
- **Freeze checks:** `python scripts/ci_freeze_check.py`.
- **Jobs:** `slurm/tasks/9*.sh`.

## 10. Next bottleneck

The behaviour GRPO taught is a procedure bound to the surface it was trained on: tool names, step order, workflow position. The next experiment that would move the science is training across tool surfaces and workflows (randomised tool names and step orders), registered against S4-EVAL's G1 and G3 splits. Making reads cost something would also make "verification" more than one unconditional read.

## Completed / attempted but failed / untested / future

- **Completed:** novelty audit; FCA implementation and development test; S4-EVAL; post-results review and corrections; product dashboard update; paper rewrite.
- **Attempted but failed:** FCA (kill criterion); RSC robustness to renamed tools and to a new workflow step.
- **Untested:** a read cost; sampled (T > 0) evaluation; a second base model; dynamic sampling or oversampling baselines; SFT on recoveries.
- **Future:** surface-randomised training; a second model family; real-system replays.
