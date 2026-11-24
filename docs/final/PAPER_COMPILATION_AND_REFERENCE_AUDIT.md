# Paper v3: compilation and reference audit (2026-11-24)

## Build
- **Files:** `paper/main_v3.tex`, `paper/appendix_v3.tex`, `paper/refs_v3.bib`.
- **Untouched:** `main.tex` and `main_v2.tex`.
- **Build sequence:** `pdflatex → bibtex → pdflatex × 2`. Run locally: there is no TeX on the cluster, and the build
  takes seconds.
- **Result:** 7 pages; 0 undefined references or citations; 0 overfull boxes after fixes.
- **Main text** (Introduction to Discussion) ends on page 5, followed by Limitations, Ethics and References. It is
  under the 6-page limit with room for the AD-C1 paragraph and one figure.
- **Anonymised** (`\usepackage[review]{emnlp2023}`, "Anonymous submission"). main_v2 carries the author's name and
  e-mail; do not submit it.

## Open placeholders (`\pend{}`)
1. **AD-C1 outcome:** abstract, §5 and Appendix AD-C1. Fill from the lead engineer's c1_stats output, and from
   `docs/research/AD_C1_ANALYSIS.md` for the theory comparison.

## Reference audit (refs_v3.bib, 16 entries)

| Key | Title / authors / year checked against | Status | Statement it supports in v3 | Supported? |
|---|---|---|---|---|
| liu2025drgrpo | arXiv abs page (8 authors, v2 Oct 2025, no venue) | ✓ | std causes difficulty bias; std removal | yes [std] (claim matches the paper's main thesis; equations not re-read this session) |
| bereket2025uncalibrated | arXiv abs (2 authors) | ✓ | std miscalibrates stochastic outcomes | yes [eq] |
| davis2025objective | arXiv HTML (2 authors) | ✓ | GRPO ≈ monotone (arcsin√) transform of per-prompt success, binary rewards | yes [eq] |
| bay2026identity | arXiv abs (2 authors, 30 Jun 2026) | ✓ | std identity for binary rewards | yes [eq, prev. session] |
| wang2026darkroom | arXiv HTML (1 author, 23 Jul 2026) | ✓ | z-score invariance to the shaping coefficient in all-fail groups | yes (Prop. 1 read) |
| sakib2026luck | arXiv abs (Nazmus Sakib, 21 Sep 2026) | ✓ | paired noise schedules within a group | yes [abs] |
| ding2026calibrate | arXiv HTML (3 authors) | ✓ | GRPO collapses to a static always-test policy across cost ratios; group construction not reported | yes (RL section read) |
| zhu2025stratified | arXiv abs (5 authors) | ✓ | cited in the appendix/related work only | — |
| feng2025gigpo | arXiv/NeurIPS listing (4 authors, NeurIPS 2025) | ✓ | not cited in the main text | — |
| liu2026gdpo | arXiv abs (13 authors; NVIDIA tech report) | ✓ | per-component normalisation | yes [eq, prev.] |
| yu2025dapo | arXiv abs (first 5 of 35 authors; "and others") | ✓ partial | dynamic sampling | yes [std] |
| ahmadian2024back | arXiv abs (8 authors) | ✓ | RLOO | yes [std] |
| mroueh2025grpo | arXiv abs (1 author) | ✓ | not cited in the main text | — |
| schulman2017ppo | [std] | not re-checked | PPO | yes [std] |
| howard1966information | [std]: IEEE Trans. SSC 2(1):22–26 | not re-checked this session | VOI | yes [std] |

**Corrections relative to refs_v2.bib:**
- liu2025drgrpo now lists all 8 authors.
- GDPO's title is completed ("…for Multi-reward RL Optimization"), with all 13 authors.
- Ding et al. now have first names and the full title.
- Sakib's first name is added.

## Numbers
Every number in main_v3 comes from one of these files:
- `results/theory/scan.json`
- `results/theory/th1_heldout_summary.json`
- `results/theory/th1_heldout_posthoc_h6.json` (labelled post hoc)
- `results/research/llm_mechanism.json`
- `results/p1/p1_summary.json`
- `results/j2/j2_summary.json`

The mapping is in `docs/final/RESEARCH_DECISION.md` §N. The v3 numbers are **not** yet in `claims/claims.yaml` or
the canonical pipeline. The lead engineer owns those, and adding them is a listed decision.
