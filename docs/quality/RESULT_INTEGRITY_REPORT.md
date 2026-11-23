# Result integrity report

2026-11-24, branch `quality-upgrade`. Generated surfaces, their sources and the checks that keep them in step.

## 1. One source of truth: `actionproof/reporting/`

| Module | Output | What it guarantees |
|---|---|---|
| `canonical.py` | `results/canonical/results.json` | Every ledger number (316) re-read from its result file, with source path and key, claim, registration, evidence label and interval metadata (or the explicit "not_recorded"). It records the sha256 of every source and its own content hash. **Today: 316 numbers, 0 mismatches, 0 read errors** |
| `freshness.py` | exit code and report | A generated surface is stale if any file its builder reads is newer than it. The canonical file is stale if any source hash changed |
| `consistency.py` | `results/canonical/consistency.json` | The ledger's token rule applied to every published surface, as a report |
| `coverage.py` | used by `scripts/p2_stats.py` | No statistic on an incomplete evaluation matrix |

Tests: `tests/reporting/test_reporting.py` and `tests/p2/test_p2.py`.

**Evidence labels** (`canonical.label`). The labels are mutually exclusive, and each displayed number carries one:

| Label | Count | Rule |
|---|---|---|
| confirmatory | 211 numbers | registered and supported |
| unsupported | 34 | registered; rule not met |
| failed | 10 | candidate dropped by its own criterion |
| exploratory | 16 | registered after results, or descriptive by registration |
| descriptive | 45 | no registration |
| development, incomplete, future | — | used when they apply |

The registered studies P1, P2 and AD-C1 are listed as `incomplete` until their registered summary file exists.

## 2. Surfaces audited

| Surface | Before today | After |
|---|---|---|
| `demo/static/index.html` (public site) | Built 2026-11-23 22:30, older than one of its sources (`scripts/site_upgrade.py`) | Rebuilt; fresh. Sections 06–09 read result files directly; section 09 (P1) shows "running" until `results/p1/p1_summary.json` exists |
| `demo/data.json` (Study 1 episodes) | Older than a source it reads (`results/judge/aya32b__summary.json`) | Rebuilt. Content identical except its git stamp, so it was stale by date only |
| `product/data/dashboard.json` (HF Space research tab) | Fresh by date, but covered **4 of 28** claim registrations: no R1, J1, E1/E2, C5, G, AD, ID1, SR1, S3 addendum 2/4, A-series | Now carries an **evidence ledger**: all 48 claims with label, registration and source files, the running studies, the canonical hash and a freshness stamp. Rendered in the Space's "Research dashboard" tab (`product/render.py::evidence_html`) |
| `product/data/replays.json` | Fresh | Fresh. S3 confirmatory replays, recorded, labelled as such in the app |
| `docs/figures/figures_data.json` | Flagged only because a glob matched directories (fixed in the checker) | Fresh |
| `docs/final/FINAL_CLAIMS.md` | Hand-written, last edited 2026-11-23 10:29, **before** the R1, J1, C5, G, AD and P1 results | Superseded for status by the canonical file and `docs/upgrade/FINAL_REVIEW.md`. Its numbers are in the consistency report (§3) |

## 3. Numbers not tied to a result file

From `consistency.py`, distinct result-like numbers with no ledger entry:

| Surface | Enforced by `s2_numcheck` | Result tokens | Unverified distinct |
|---|---|---|---|
| README.md | yes | 82 | 1 (`1.0`, a version-like token) |
| paper/main.tex, appendix.tex | yes | 134 | 0 |
| docs/stage2/VIDEO_SCRIPT.md, PRODUCT.md | yes | 16 | 0 |
| **docs/submission/WRITEUP.md** | **no** | 139 | **58** |
| docs/final/FINAL_CLAIMS.md | no | 125 | 23 |
| docs/upgrade/FINAL_REVIEW.md | no | 31 | 12 |
| demo/site_template.html (hand-typed page text) | no | 79 | 34 |
| docs/upgrade/JUDGES.md, CAPABILITY_RETENTION.md, ROI.md | no; generated from result files by `scripts/render_upgrade_reports.py` and `roi_model.py` | 595 | 268 (file-backed, but not in the ledger) |
| docs/ARTICLE_DRAFT.md | no; superseded draft | 573 | 203 |

**What the write-up's unverified numbers are.** Real values copied from result files:
- R1 retention deltas (for example +0.005, −0.038);
- J1 per-judge accuracies (0.576, 0.373, 0.514, 0.300);
- AD development VSC (0.707, 0.833, 0.867).

They are correct today. Because no entry pins them, a later change in a result file would not be caught.

**Site template.** 34 hand-typed numbers in page text: Study 1 percentages, the BFCL case count (3,491), the model size (3.35B), and others. They are the "manually copied score" risk the brief names.

## 4. Remaining actions

- **Before submission.** Add `docs/submission/WRITEUP.md` to the ledger's enforced documents, with entries for every number it prints, including the P1 and P2 numbers when they report. Then `s2_numcheck` fails on any untied number.
- **The site template's hand-typed numbers.** Either move them into the data the page reads from result files, or give each a ledger entry. Study 1 figures come from `docs/figures/figures_data.json`, which is generated.
- **`docs/ARTICLE_DRAFT.md`** is superseded by the write-up. Mark it as such, so it is not mistaken for a current surface.
