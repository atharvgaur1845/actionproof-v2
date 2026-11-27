# Official rules and compliance audit (prompt §10)

2026-11-27, 10:25 IST. Source: https://hackathon.lexsi.ai/, re-read today (rules, FAQ, Structure, Prizes).
Each row is checked against an artefact or file, not a README sentence.

**Deadline.** Finalists submit a video presentation or slides by **Nov 28, 11:59 PM AoE** (= Nov 29, 17:29 IST),
through a separate Stage 2 form mailed to finalists. This sprint freezes earlier: Nov 28, 23:59 IST.

**Inconsistencies on the official page, to confirm with the organisers if they matter:**
- The FAQ gives the hackathon as Nov 1–22, while other sections say Nov 30.
- The credit window is Nov 16–22 under Prizes and Nov 19–28 under Structure.
- "Public deployment by the submission deadline" does not say which deadline.

| Requirement (official wording, condensed) | Status | Evidence |
|---|---|---|
| Core model: a dense open-weights Cohere Labs model (Tiny Aya Base/Global/Fire/Water/Earth, Aya Expanse 8B/32B) | **Met** | Every trained agent starts from `CohereLabs/tiny-aya-global` (`release/ARTIFACT_MANIFEST.json`, `base_model` on each card). P5 uses Tiny Aya Fire, Water, Earth and Aya Expanse 8B. Prompted baselines use Aya Expanse 8B and 32B |
| Other models only supplementary (teacher, validation, judge) | **Met** | Qwen3-235B-A22B and gpt-oss-120b appear only as labelled supplementary baselines (`PREREGISTRATION_B1_ADDENDUM1.md`; README "Two large open-weight models…"). LLM judges (J1) are evaluation instruments only (WRITEUP §15). No external model generates training data: SEL1's demonstrations come from the scripted B5 policy (`actionproof/sel1/build.py`) |
| Track 3 toolkits: AgentTune, AuditKIT (the two-repo minimum does not apply to Track 3) | **Met, in executed code** | AgentTune: GRPO trainer `TrlAgenticGrpo` (`actionproof/train/grpo_proc.py:45`), rollout engine and tool-call parser (`actionproof/rollout/episode.py:23-25`, `actionproof/rollout/agenttune_rollout.py:78`). AuditKIT: TaskCompletion judge on P1 test episodes (`actionproof/eval/auditkit_bridge.py`, outputs in `results/p1/auditkit/`). Revisions pinned in the manifest (`environment`) |
| Developed during the hackathon; declaration of earlier or outside work | **Met; one gap fixed below** | WRITEUP §30 states that nothing predates Nov 1, 2026. The declaration did not yet name the Nov 26–28 work (TXN1, P4, P5, SEL1, the live Space); it is added to §30 at the freeze |
| Public deployment | **Met** | Live Space https://huggingface.co/spaces/atharvgaur18/actionproof-live (real inference; commit cf90ad3, API-tested today, `docs/submission/DEMO_RUN_OF_SHOW.md`). Replay Space https://huggingface.co/spaces/atharvgaur18/actionproof (static, commit 83df86f, served page sha256 `73579db9…` = the committed `demo/static/index.html`) |
| Public, un-gated HF collections for models, adapters, datasets, logs and traces | **Met (collection created today)** | https://huggingface.co/collections/atharvgaur18/actionproof-final-round-stage-2-6b090c4faec34bf64ccdb0f0: 10 items, public to an anonymous request. Every listed repo is public and un-gated anonymously (`release/ARTIFACT_MANIFEST.json`). The base model is gated by Cohere Labs (third party); our repos are not. Three earlier collections hold the Stage 1 and 2 artefacts |
| Write-up (HF article or GitHub README) | **Met, with an author action** | `docs/submission/WRITEUP.md` (full) and `README.md` (landing page). The GitHub branch is not pushed yet: that needs the author's approval |
| GitHub repo with code and reproducibility notes | **Met once pushed** | `README.md` "Reproduce"; `scripts/smoke_check.py`; `scripts/verify_numbers.sh`; `docs/submission/REPRODUCIBILITY_FINAL.md` |
| Abstract under 500 words | **Met** | WRITEUP §1: 499 words, heading included (counted by script) |
| Required write-up sections | **Met** | Title and claim (top); tracks and problem statement (table); abstract §1; problem and approach §4–§5; kits §10; base model and licence §11; data and licences §12; splits and leak checks §12; metrics and rationale §14; method and experiments §1a, §6, §17–§19; before/after §2; limitations §25; planned work §27; artefacts §28; reproduction §29; declaration §30; teammates (solo; "Team" row) |
| Before/after with more than one metric, and why each | **Met** | VSC/VTC, false completion, duplicate mutation, unnecessary retry, escalation, unsafe, selective risk (WRITEUP §14; `actionproof/txn/truth.py`). Same-base before/after: §2 |
| "Nothing else got worse" (optional) | **Met** | R1/R1-T capability retention (WRITEUP §16); P4 H3 and SEL1 H4 non-inferiority on the canonical tests |
| Track 3 metrics: task success, tool-call accuracy, latency | **Partly met** | Task success: VSC and VTC throughout. Latency: `results/latency_single.json` (Stage 2 models; not re-measured for D). **Tool-call accuracy is not reported as a named metric.** Unparseable turns end an episode (`context_ended`, and calls per success, in every `results/p1/test/*.summary.json`). Disclosed in the freeze checklist |
| ROI with GPU hours and cost assumptions (optional, preferred) | **Met** | WRITEUP §24 and `docs/upgrade/ROI.md`; `results/impact/impact_txn.json` (rates measured, dollars projected, assumptions stated) |
| Video or slides | **Slides met; video is an author action** | Deck https://claude.ai/artifact/N5NxS6YHhbaGqGSUMqeYbW (exports to PDF or PPTX; the author must share or export it). Video: `docs/submission/VIDEO_SCRIPT.md` and `VIDEO_CAPTIONS.srt` are ready; the recording is the author's |
| Licences for the base model, data and benchmarks | **Met** | WRITEUP §11–§12. Models CC-BY-NC-4.0 (derived from Cohere's licence); code Apache-2.0; manifest `licence` |
| Attribution; LLM-use statement (optional) | **Met; share added** | WRITEUP §31–§33. The optional "approximate AI-generated share" is added to §33 at the freeze |
| Originality / plagiarism | **No issue found** | Competitor repositories were read, never copied or run (`research_audit/COMPETITOR_FORENSICS_TOP20.md`); the closest prior designs are cited (`research_audit/NOVELTY_POSITIONING_FINAL.md`) |

**Author-only actions this audit cannot complete:**
1. Approve pushing the branch to GitHub, and its merge into `submission-candidate`.
2. Record the video.
3. Export or share the deck.
4. Set the live Space's `HF_TOKEN` secret for the gated base model.
5. Submit the Stage 2 form.
