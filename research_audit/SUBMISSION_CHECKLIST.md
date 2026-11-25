# Submission checklist

2026-11-25. Branch `submission-candidate`. Brief deadline: Nov 28, 23:59 IST. The author earlier said there were about 40 hours left from Nov 24 07:45. **The author must confirm the deadline on the portal.**

## Integrity (checked by commands, 2026-11-25)

- [x] Number check: `python scripts/s2_numcheck.py`. 628 numbers in 10 documents (README, write-up, paper v1/v2/v3 with appendices, video script, product), 0 failures.
- [x] Generated surfaces fresh: `python -m actionproof.reporting.freshness`. Site, demo data, dashboard, replays, figures and canonical file.
- [x] Registrations unchanged: `freeze_p1/p2/p3 --check` and `scripts/ad/freeze_c1.py --check` all OK. The earlier studies' freezes are audited in `docs/quality/RESULT_INTEGRITY_REPORT.md`.
- [x] Headline numbers recomputed from raw episodes by a separate pathway: `research_audit/CLAIM_VERIFICATION.md`. All recomputed headlines pass; no headline failed.
- [x] No duplicate episodes; every result file covers its frozen split exactly (242 files).
- [x] No fault word appears in any task prompt. One unused split overlaps a training pool (`c1_pp` / `cw_train`; `DATASET_EXPANSION.md` §2).
- [x] `c1_pp` removed from the write-up's confirmatory splits; the overlap is stated.
- [x] Read-back wording made exact in the write-up (`CLAIM_VERIFICATION.md` note a).

## Results still running (fill when they report; never before)

- [ ] **P3** (`PREREGISTRATION_P3.md`, tasks 1610–1613). Add `results/p3/p3_summary.json` to the ledger, write-up, README, paper and site.
- [ ] **Judge study** (tasks 1600, 1601). Run `python evaluation/judge_next/analyze.py`, then fill `research_audit/JUDGE_RELIABILITY.md`.
- [ ] **P2 judge committee** (job 467654, finished). Pull its output and add it as a descriptive result.

## Deliverables

- [x] `research_audit/`:
  - BRANCH_AUDIT, MASTER_EXPERIMENT_LEDGER, CLAIM_VERIFICATION, SCIENTIFIC_STORY;
  - ALGORITHM_DISCOVERY, DATASET_EXPANSION, EXTERNAL_BENCHMARKS, INTERPRETABILITY_AUDIT;
  - EXPERIMENT_PRIORITY, this file.
- [ ] `research_audit/JUDGE_RELIABILITY.md` and `results/judge_next/{configs,metrics,confusion_matrices,per_fault,report.md}` (after tasks 1600 and 1601)
- [ ] `research_audit/FINAL_RECOMMENDATIONS.md` (after P3)
- [x] Paper v3 builds, anonymised, 7 pages (`paper/main_v3.pdf`)
- [x] Site redeployed on Nov 24 with sections 09–10; redeploy again after P3 and the judge results
- [x] Write-up `docs/submission/WRITEUP.md` enforced by the number check

## Author-only (cannot be done by the agent)

- [ ] Confirm the deadline on the portal.
- [ ] The §30 declaration of work done outside the hackathon, and team contributions (§31).
- [ ] Human labels: the 76 J1 calibration cases, or keep every judge result explicitly model-vs-verifier (the current wording).
- [ ] Title and thesis of paper v3 (options in `docs/final/RESEARCH_DECISION.md`).
- [ ] Whether to fast-forward `main` to `submission-candidate`, which changes the public default branch.
- [ ] Publishing the ~760 MB of P1/P2 episode traces to the HF dataset.
- [ ] Final redeploy of the HF Space after the last results.
