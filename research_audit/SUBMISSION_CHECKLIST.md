> **Superseded (2026-11-27, Stage 2 sprint).** This file is kept as history (last updated 2026-11-25) and is not current. The current source is `research_audit/FINAL_FREEZE_CHECKLIST.md`. See `research_audit/STALE_SURFACES_AUDIT.md`.

# Submission checklist

2026-11-25. Branch `submission-candidate`. **Deadline confirmed by the author: Nov 28, 2026, 23:59 IST.**

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

- [x] **P3** reported: not supported (−0.007); in ledger, write-up, README, paper, site.
- [x] **Judge study** analysed: `JUDGE_RELIABILITY.md`, `results/judge_next/`.
- [x] P2 judge committee added (descriptive, `results/p2/committee_p2.json`; ledger claim P2-COMMITTEE).

## Deliverables

- [x] `research_audit/`:
  - BRANCH_AUDIT, MASTER_EXPERIMENT_LEDGER, CLAIM_VERIFICATION, SCIENTIFIC_STORY;
  - ALGORITHM_DISCOVERY, DATASET_EXPANSION, EXTERNAL_BENCHMARKS, INTERPRETABILITY_AUDIT;
  - EXPERIMENT_PRIORITY, this file.
- [x] `research_audit/JUDGE_RELIABILITY.md` and `results/judge_next/`
- [x] `research_audit/FINAL_RECOMMENDATIONS.md` (filled)
- [x] Paper v3 builds, anonymised, 7 pages (`paper/main_v3.pdf`)
- [x] Site redeployed on Nov 24 with sections 09–10; redeploy again after P3 and the judge results
- [x] Write-up `docs/submission/WRITEUP.md` enforced by the number check

## Author-only (cannot be done by the agent)

- [x] Deadline confirmed by the author (Nov 28, 23:59 IST).
- [ ] The §30 declaration of work done outside the hackathon, and team contributions (§31).
- [ ] Human labels: the 76 J1 calibration cases, or keep every judge result explicitly model-vs-verifier (the current wording).
- [ ] Title and thesis of paper v3 (options in `docs/final/RESEARCH_DECISION.md`).
- [ ] Whether to fast-forward `main` to `submission-candidate`, which changes the public default branch.
- [ ] Publishing the ~760 MB of P1/P2 episode traces to the HF dataset.
- [x] HF Space redeployed Nov 25 with P3 and the 2x2 (Space commit 644c862).
