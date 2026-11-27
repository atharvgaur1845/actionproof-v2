# Final freeze checklist (prompt §14)

**State** (written 2026-11-27, 11:00 IST):
- branch `stage2-superiority-sprint`, local only; nothing has been pushed or merged;
- conservative freeze Nov 28, 23:59 IST; official deadline Nov 28, 11:59 PM AoE = Nov 29, 17:29 IST.

Status: ✅ met · ⚠️ met with a disclosed limit · ❌ not met · 👤 the author's action.

## Competitive understanding

| Criterion | Status | Evidence |
|---|---|---|
| All 20 competitor repos audited, or labelled not inspectable | ✅ | `research_audit/COMPETITOR_FORENSICS_TOP20.md`, `research_audit/competitors/` |
| At least five lessons from the strongest projects turned into improvements | ✅ | From `COMPETITIVE_THREAT_RANKING.md` "Which advantages can we match": (1) a rendered six-slide deck and a video-ready run of show (01, 04); (2) one thesis across every surface (top three); (3) a numbered judge path in the README and the Space (06); (4) a 15–30 s no-GPU smoke check beside the full route (02, 06); (5) a Hub revision manifest (06); (6) a live demo that degrades gracefully, with labelled recorded runs (04); (7) an always-escalate reference next to every escalation metric, and a selective-escalation experiment (02); (8) an impact graphic (04) |
| Threat ranking from live evidence | ✅ | `research_audit/COMPETITIVE_THREAT_RANKING.md` (Hub API status of every Space) |

## Research quality

| Criterion | Status | Evidence |
|---|---|---|
| A one-sentence thesis backed by named result files | ✅ | README "In one paragraph"; write-up one-line claim; `docs/submission/FINAL_STORY_DECISION.md` §2 and §4 |
| Every claim carries an evidence status and a scope | ✅ | Status legend in the README; `claims/claims.yaml` (82 claims with status, 866 numbers) |
| No unsupported optimizer, generalisation, circuit, safety or ROI claim | ⚠️ | Removed or scoped after the hostile review (`research_audit/FINAL_REDTEAM_REVIEW.md` rows 5, 13, 15, 24–27). ROI: projections, both units, assumptions stated |
| Positive and negative results both visible | ✅ | README "What does not work" and "TXN v1: limits"; write-up §22 |
| Key baselines fair and fully specified | ⚠️ | B0–B4 and A1–A6 in `actionproof/txn/policies.py`; post-hoc read-free references FF and BD added. B3 cannot see injected duplicates, so the H3 headline is shown with its origin split (TXN1 addendum 3) |
| Any new confirmatory experiment registered and frozen before its outcome | ✅ | SEL1: `PREREGISTRATION_SEL1.md` at 795e7cd, 05:39 IST; data hashes 05:42 (`results/sel1_hashes.json`, recovered from the cluster); first training 05:44 |

## Product and presentation

| Criterion | Status | Evidence |
|---|---|---|
| Problem clear in 15 s; core demo in 60–90 s | ⚠️ | README hook and judge path. Judge mode is preselected; a run takes about 70 s once warm, longer from cold (not measured) |
| Live model, scripted reference and recorded replay never conflated | ✅ | Banners `LIVE MODEL RUN`, `SCRIPTED`, `RECORDED MODEL RUN`, `LIVE RUN FAILED`, checked through the API on staging and live |
| The state mismatch visible in the interface | ✅ | The post-run reveal: database rows and the verdict per step |
| Six slides and a 2:30–2:50 video telling the write-up's story | 👤 | Deck finished (artifact version 1795756118-dd8d; text in `docs/submission/SLIDES.md`). Video script and captions are 2:45 on the same timeline. **The video is not recorded** |
| A figure that makes impact and remaining risk clear | ✅ | Deck slide 6 (impact bars in both units) and slide 5 (what fails) |

## Reproducibility

| Criterion | Status | Evidence |
|---|---|---|
| A fresh clone runs the CPU verification path | ✅ | `docs/submission/REPRODUCIBILITY_FINAL.md`: two clean-room runs, the final at 319a383. The released SEL1 training data match their pre-training hashes (task 1861) |
| The full trace-backed analysis has one tested command | ✅ | `bash scripts/verify_numbers.sh` (passed in both runs) |
| Frozen hashes and model revisions public and documented | ✅ | `release/ARTIFACT_MANIFEST.json`; `REPRODUCIBILITY_FINAL.md` |
| Model cards, dataset card, code, traces and demo URLs checked | ⚠️ | 14 URLs return HTTP 200 anonymously. Cards match the repo byte for byte. **GitHub serves an older state until the push** |
| The claims checker passes with zero unexplained failures | ✅ | `python scripts/s2_numcheck.py`: 866 numbers, 14 documents, 0 failures |

## Final adversarial review

| Criterion | Status | Evidence |
|---|---|---|
| No unresolved critical blocker | ⚠️ | Every blocker this session can fix is fixed or disclosed (`FINAL_REDTEAM_REVIEW.md`). Two blockers need the author: push and merge (row 9); the deck and video (row 10) |
| Remaining major limits disclosed prominently | ✅ | README one-paragraph summary, "What does not work", "TXN v1: limits"; the abstract |
| No claim to beat every competitor without evidence | ✅ | None is made |

## Official rules

`research_audit/COMPLIANCE_AUDIT.md`. Every requirement is met or listed. Two are partial:
- tool-call accuracy is not a named metric;
- the video is not recorded.

## 👤 Author-only actions, in order

1. **Approve the push.** Then push `stage2-superiority-sprint`, merge it into `submission-candidate`, and point the
   default branch (or the README link) at it. Nothing has been pushed; the rules of this sprint require approval.
2. **Record the video** from `docs/submission/VIDEO_SCRIPT.md`, with the captions
   `docs/submission/VIDEO_CAPTIONS.srt`. Keep every banner and label on screen.
3. **Share or export the deck** (https://claude.ai/artifact/N5NxS6YHhbaGqGSUMqeYbW → PDF or PPTX). Add the export to
   the submission form, and to the repo if wanted.
4. **Set the live Space's `HF_TOKEN` secret** (a read token that has accepted Tiny Aya's terms), so the base-model
   options run live instead of falling back to the recording.
5. **Confirm the LLM-use share** in write-up §33 (about 95% of code, about 90% of text). Edit it if your estimate
   differs.
6. **Check the layout in a browser:** the Space at 1280 px and on mobile, and the deck slides for overflow. The
   headless browser here does not start.
7. **Submit the Stage 2 form** with the links below.

## Final references

| | |
|---|---|
| HEAD of `stage2-superiority-sprint` | The commit that adds this line; `git log -1`. Code and results verified in the clean room at 319a383; later commits change only documents and the manifest (`git diff --stat 319a383..HEAD`) |
| Live Space | https://huggingface.co/spaces/atharvgaur18/actionproof-live |
| Replay Space | https://huggingface.co/spaces/atharvgaur18/actionproof |
| Collection | https://huggingface.co/collections/atharvgaur18/actionproof-final-round-stage-2-6b090c4faec34bf64ccdb0f0 |
| Recommended model | https://huggingface.co/atharvgaur18/actionproof-agent-d |
| Traces | https://huggingface.co/datasets/atharvgaur18/actionproof-traces |
| Deck | https://claude.ai/artifact/N5NxS6YHhbaGqGSUMqeYbW (private until shared) |
| Code | https://github.com/atharvgaur1845/actionproof-v2 (after the push) |
