# Deliverables checklist (research upgrade, Nov 23)

**Status legend:**
- **done**: the artefact exists and is committed;
- **running**: registered, with jobs in progress;
- **pending**: waits on a result or on the author.

| # | Deliverable | Artefact | Status |
|---|---|---|---|
| A | Scientific audit and gap report | `docs/upgrade/AUDIT.md`, `docs/upgrade/INVENTORY.md`, `results/inventory.json` | done |
| B | Literature review and novelty matrix | `docs/research/NOVELTY_AUDIT.md` (84 entries), `NOVELTY_MATRIX.md`; extended by the discovery agent (`docs/research/algorithm_discovery/`) | done (existing); extension running |
| C | At least three algorithm candidates and a selection decision | FCA (failed), C5 (dropped, diagnosed), G (seed 1 re-running), AD: no-std GRPO, AWRB, AWRB-no-std, D2 EMA scale, D3 per-read cost (`docs/research/algorithm_discovery/CANDIDATE_RANKING.md`) | development done for C5 and AD-D1; AD-C1 confirmatory overnight |
| D | Selected candidates implemented and tested | C5/G (`actionproof/stage4/*c5g*`); AD estimators with 17 enumeration tests (`tests/ad/`) | C5 dropped; AWRB passes development gates; confirmatory pending |
| E | Multi-seed baselines and ablations | A12, S3 and addenda, S4, SR1, E1/E2 (`docs/final/FINAL_CLAIMS.md`) | done |
| F | Judge committee and human calibration | `PREREGISTRATION_J1.md`, addendum 1, `docs/upgrade/JUDGES.md` | done ($3.20); **human labels: author** (`label.html`, 76 cases) |
| G | External capability benchmarks before and after | `docs/upgrade/CAPABILITY_RETENTION.md` (R1 and R1-T) | done |
| H | Operational impact and ROI | `docs/upgrade/ROI.md`, `results/roi/roi_model.json` | done |
| I | Lexsi write-up | `docs/submission/WRITEUP.md` (all mandatory fields), `docs/submission/TRACES.md` | complete draft; τ-bench, G and AD-C1 results still to add |
| J | Website and demo | sections 06–08 in `demo/site_template.html`, calculator `demo/roi.js` (tested) | built locally; **deploy needs the author's approval** |
| K | Reproducible artefacts | `docs/upgrade/TOOLKITS.md`, freeze scripts, write-up §29; R1 and J1 outputs to publish | partly; publishing R1 and J1 outputs needs approval |
| L | Final scientific review | `docs/upgrade/FINAL_REVIEW.md` | draft done; updated when the algorithm programme reports |

## Author actions

1. **Label the 76 calibration cases.** Open `evaluation/llm_judge_committee/label.html` in a browser, then **Export**, and save the file as `evaluation/llm_judge_committee/data/human_labels.json`.
2. **Confirm** the declaration of work done outside the hackathon (write-up §30).
3. **Approve** redeploying the site to the HF Space, and publishing the R1 and J1 outputs to the HF collection, once the results are in.
