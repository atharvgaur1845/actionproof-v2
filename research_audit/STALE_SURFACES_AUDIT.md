# Stale-surface audit

2026-11-27. Base `7ae863f`. "Last relevant update" is the date of the last git commit touching the file. A surface
is **stale** if it contradicts or omits the current final-round results (TXN1, P4, P5, the live Space), or if it
describes an artefact that no longer exists.

The canonical surfaces after this sprint:
- `README.md`: the landing page;
- `docs/submission/WRITEUP.md`: the full write-up;
- `docs/submission/SLIDES.md`, plus the rendered deck;
- `docs/submission/VIDEO_SCRIPT.md` and `VIDEO_CAPTIONS.srt`;
- `docs/submission/DEMO_RUN_OF_SHOW.md`;
- `claims/claims.yaml`: numbers;
- `release/ARTIFACT_MANIFEST.json`: artefacts.

| Path | Last update | Status | Conflict | Canonical replacement | Action |
|---|---|---|---|---|---|
| `research_audit/FINAL_RECOMMENDATIONS.md` | 11-25 22:39 | stale | Predates TXN1/P4/P5; recommends work that is now done or superseded | `docs/submission/FINAL_STORY_DECISION.md` | Add a "superseded" banner pointing to the replacement; keep as history |
| `research_audit/SUBMISSION_CHECKLIST.md` | 11-25 22:39 | stale | §30 and §31 shown as open (`[ ]`); WRITEUP §30–34 now exist (§30 confirmed by the author, commit `7ae863f`) | `research_audit/FINAL_FREEZE_CHECKLIST.md` | Superseded banner |
| `research_audit/SCIENTIFIC_STORY.md` | 11-25 07:55 | stale | Story without TXN1/P4/P5 | `docs/submission/FINAL_STORY_DECISION.md` | Superseded banner |
| `research_audit/MASTER_EXPERIMENT_LEDGER.md` | 11-25 07:43 | stale | No TXN1/P4/P5/B1-API/R7 rows | `claims/claims.yaml` + `docs/upgrade/INVENTORY.md` | Superseded banner (ledger of Nov 25) |
| `research_audit/CLAIM_VERIFICATION.md` | 11-25 07:53 | stale | Verifies Nov 25 claims only | `claims/claims.yaml` + `s2_numcheck` | Superseded banner |
| `research_audit/BRANCH_AUDIT.md` | 11-25 08:52 | stale | Branch heads have moved; all branches are now merged | `research_audit/CURRENT_BRANCH_FORENSIC_REVIEW.md` | Superseded banner |
| `research_audit/INTERPRETABILITY_AUDIT.md` | 11-25 08:07 | current (scope) | A15/A16 unchanged since | — | Leave |
| `research_audit/JUDGE_RELIABILITY.md` | 11-25 22:36 | current (scope) | No later judge work | — | Leave |
| `docs/final/FINAL_CLAIMS.md` | 11-23 10:29 | stale | Nov 22 claims summary; says it is "the source of truth for every claim on a slide"; no P1–P5/TXN | `claims/claims.yaml` | Superseded banner; it must not claim to be the source of truth |
| `docs/final/DEMO_RUN_OF_SHOW.md` | 11-22 01:34 | stale | Describes a static-site-only demo with "no live model inference" built from `stage4` | `docs/submission/DEMO_RUN_OF_SHOW.md` (new) | Superseded banner |
| `demo/README.md` | 11-13 12:58 | stale | Front matter of the *old* Gradio demo (`sdk: gradio`, `app_file: app.py`); the replay Space is now static (`demo/static/`) and the live one is `space_live/` | `space_live/README.md` (live), `demo/static/README.md` (replay) | Replace with a short pointer README |
| `release/README_SECTION.md` | 11-26 04:13 | stale (partial) | No Dw_aug row (`actionproof-warmstart-dwaug`, released Nov 26) | `release/ARTIFACT_MANIFEST.json` and the README table | Add the Dw_aug row |
| `README.md`, "Deliverables" table | 11-27 | stale (partial) | Points to `docs/ARTICLE_DRAFT.md` as the write-up (canonical is `docs/submission/WRITEUP.md`); "29 LoRA adapters" collection; the repository map says `demo/` is a "Gradio Space (live ZeroGPU runs…)" | this README, restructured | Rewrite the section |
| `README.md`, "Reproduce §2" | 11-27 | stale (partial) | "no CI is configured" (CI exists); clone check dated Nov 15 | the new "Reproduce" section | Rewrite |
| `README.md`, Stages 1–4 history | 11-27 | current but misplaced | About 150 lines of history between the headline and the reproduce section | `docs/HISTORY.md` (new) | Move; keep one link |
| `docs/ARTICLE_DRAFT.md` | 11-24 07:12 | stale | Pre-final-round article | `docs/submission/WRITEUP.md` | Superseded banner |
| `docs/ABSTRACT.md` | 11-15 21:59 | stale | Stage 1 abstract | WRITEUP §1 | Superseded banner |
| `docs/HF_ARTICLE.md` | 11-15 21:59 | stale | Stage 1 article | WRITEUP | Superseded banner |
| `docs/submission/TRACES.md` | 11-23 11:42 | stale (partial) | Lists traces before TXN/P4/P5 | `release/dataset_card.md` + manifest | Superseded banner |
| `docs/CONSISTENCY_AUDIT.md` | 11-22 00:10 | historical | Rounds 1–2 of wording fixes | — | Leave (history) |
| `docs/quality/RESULT_INTEGRITY_REPORT.md` | 11-24 01:48 | historical | P1/P2 integrity | — | Leave (history) |
| `docs/submission/WRITEUP.md` | 11-27 03:32 | current, thesis conflict | Title and abstract lead with Stage 2's reward-information story; the final-round results sit in §1a | — | Re-lead the title and abstract with the decided spine (FINAL_STORY_DECISION); keep the body |
| `paper/main_v3.tex` | 11-27 03:24 | current, different deliverable | A theory paper on group normalisation, not the hackathon's primary narrative | — | Label it a **companion theory paper** on the landing page and in the write-up |
| `docs/submission/SLIDES.md` | 11-27 03:24 | current | Markdown only | a rendered deck | Render |
| `docs/submission/VIDEO_SCRIPT.md` / `.srt` | 11-27 03:24 | current | Captions timed to the script, not to a cut | — | Keep; retime after the recording (author) |
| `demo/static/index.html` (replay site) | built 11-27 | current | Not scanned by numcheck (generated from result files) | — | Leave; note in the reproducibility doc |
| `docs/stage2/VIDEO_SCRIPT.md` | earlier | stale | Older script; still in numcheck's documents list | `docs/submission/VIDEO_SCRIPT.md` | Superseded banner (keep it scanned) |
