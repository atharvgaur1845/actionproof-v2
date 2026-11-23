# Deliverables checklist (research upgrade, Nov 23)

**Status legend:**
- **done**: the artefact exists and is committed;
- **running**: registered, with jobs in progress;
- **pending**: waits on a result or on the author.

| # | Deliverable | Artefact | Status |
|---|---|---|---|
| A | Scientific audit and gap report | `docs/upgrade/AUDIT.md`, `docs/upgrade/INVENTORY.md`, `results/inventory.json` | done |
| B | Literature review and novelty matrix | `docs/research/NOVELTY_AUDIT.md` (84 entries), `NOVELTY_MATRIX.md`; extended by the discovery agent (`docs/research/algorithm_discovery/`) | done (existing); extension running |
| C | At least three algorithm candidates and a selection decision | FCA (failed), C5 and G (development, running), plus the discovery agent's candidates; `docs/research/algorithm/DECISION.md` | running |
| D | Selected candidates implemented and tested | C5 and G code, tests and runs (`actionproof/stage4/*c5g*`, `tests/stage4/test_c5g.py`) | running (kill decisions due about 12:30) |
| E | Multi-seed baselines and ablations | A12, S3 and addenda, S4, SR1, E1/E2 (`docs/final/FINAL_CLAIMS.md`) | done |
| F | Judge committee and human calibration | `PREREGISTRATION_J1.md`, `evaluation/llm_judge_committee/` | running; **human labels: author** (`label.html`, 76 cases) |
| G | External capability benchmarks before and after | `PREREGISTRATION_R1.md`, `PREREGISTRATION_R1_TAU.md`, `actionproof/retention/` | running |
| H | Operational impact and ROI | `docs/upgrade/ROI.md`, `results/roi/roi_model.json` | done |
| I | Lexsi write-up | `docs/submission/WRITEUP.md` (all mandatory fields), `docs/submission/TRACES.md` | draft; R1, J1 and C5/G sections pending |
| J | Website and demo | sections 06–08 in `demo/site_template.html`, calculator `demo/roi.js` (tested) | built locally; **deploy needs the author's approval** |
| K | Reproducible artefacts | `docs/upgrade/TOOLKITS.md`, freeze scripts, write-up §29; R1 and J1 outputs to publish | partly; publishing R1 and J1 outputs needs approval |
| L | Final scientific review | after results land | pending |

## Author actions

1. **Label the 76 calibration cases.** Open `evaluation/llm_judge_committee/label.html` in a browser, then **Export**, and save the file as `evaluation/llm_judge_committee/data/human_labels.json`.
2. **Confirm** the declaration of work done outside the hackathon (write-up §30).
3. **Approve** redeploying the site to the HF Space, and publishing the R1 and J1 outputs to the HF collection, once the results are in.
