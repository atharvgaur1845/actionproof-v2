# Publication consistency report

2026-11-24. Generated checks: `scripts/s2_numcheck.py` (fails on any untied number in an enforced document) and `python -m actionproof.reporting.consistency` (reports on every surface, `results/canonical/consistency.json`).

## Enforced documents: 0 failures

The enforced documents are README.md, docs/submission/WRITEUP.md (newly enforced today), the paper (v1 and v2, main and appendix), VIDEO_SCRIPT and PRODUCT. Every result number in them resolves to a ledger entry, and each entry is re-read from its result file.

What enforcing the write-up took:
- 55 untied numbers were tied to ledger entries (J1-DESC, R1-TABLE, AD-DEV, ROI, A12-GATE0);
- constants (sample sizes, hyperparameters, arXiv ids, task amounts, a cited survey range) went into the `allow` list;
- two sentences were reworded so they carry only ledgered values ("about 80%" became the two ledgered dollar figures).

## Not enforced (reported)

| Surface | Status |
|---|---|
| demo/site_template.html | Hand-typed Stage 1 figures. Sections 06–09 read result files directly |
| docs/upgrade/JUDGES.md, CAPABILITY_RETENTION.md, ROI.md | Generated from result files by scripts. File-backed, but not every number has a ledger entry |
| docs/final/FINAL_CLAIMS.md | Superseded for status by `docs/upgrade/FINAL_REVIEW.md` and the canonical file |
| docs/ARTICLE_DRAFT.md | Marked superseded |

## Canonical file

`results/canonical/results.json` records every ledger number with its source, key, claim, registration and evidence label, plus source hashes. The freshness check reports every generated surface as fresh.
