# Stage 2 superiority sprint: progress log

Branch `stage2-superiority-sprint` (from `submission-candidate` @ `7ae863f`). Local commits only; no push.

## Phase A: competitor and branch audit (2026-11-27 05:29 IST)
- Cloned and profiled all 20 competitor repositories (read-only; no competitor code run); Hub API status for every Space and headline artefact.
- Deliverables: COMPETITOR_FORENSICS_TOP20.md, COMPETITIVE_CAPABILITY_MATRIX.md, COMPETITIVE_THREAT_RANKING.md, CURRENT_BRANCH_FORENSIC_REVIEW.md, STALE_SURFACES_AUDIT.md; raw notes and profiles in research_audit/competitors/.
- Key findings:
  - 10 of 12 competitor Spaces are static replays; ours is the only awake live Cohere-agent Space in Track 3.
  - Our weakest columns are Presentation (Markdown-only slides, no video, three theses) and Impact.
  - The open scientific flank is escalation (H5), where Escalation-Safe Aya is strongest.
- Official rules re-read: Stage 2 deadline Nov 28 23:59 AoE (= Nov 29 17:29 IST); freeze planned for Nov 28 23:59 IST; finalists judged on the video or slides; external models only supplementary.

## Phase B: research direction (2026-11-27 05:29 IST)
- NOVELTY_POSITIONING_FINAL.md:
  - the paired observation-equivalent design is not ours ("Did It Happen?", LIMBO);
  - the prior escalation work (AgentAbstain, HiL-Bench, EpiSEM-Bench, Evidence-Guided Recovery) is evaluation-only or prompt-based.
- FINAL_STORY_DECISION.md: the spine is TXN benchmark + trained recovery (C+B); supporting findings are selective escalation (D) and reward information (A); the theory paper is a labelled companion (E).
- EXPERIMENT_DECISION.md: SEL1 (selective escalation) is the one primary experiment; A (GRPO from Dw_aug) and C (gateway) are dropped; launch gate at 13:00 IST.

## Phase C: the registered experiment, SEL1 (registered 05:39 IST; summary file written 09:48 IST; logged 10:15 IST)
- `PREREGISTRATION_SEL1.md` committed (795e7cd) before any training; splits and data hashed (`results/sel1_hashes.json`).
- Tasks 1800–1812 on the a100 and h100 worker lanes. Every run finished on its first attempt; the freeze check
  passed before the analysis (task 1812).
- **Result, as registered: SEL1 supported.**
  - H1 (escalation on the locked test, unresolvable): +0.970 [+0.952, +0.987].
  - H3 (selectivity J): +0.732 [+0.625, +0.836].
  - H4 (resolvable VTC): +0.002 [−0.006, +0.010].
  - H2 (held-out compositions) passes: +0.618 [+0.486, +0.754].
- **Reported limits:**
  - On the held-out `receive` step, seed 2 escalates in 2 of 120 episodes and is unsafe in 112 (its control 83).
  - Overall txn_test VTC: 3285 against 3451 of 3960.
- **Found when the impact projection was rerun:** both SEL1 arms pass B4's overall TXN VTC (0.830 and 0.871 against
  0.864), but only on unresolvable quadruples, where they complete without the evidence. B4 completes every other task.
  "The scripted reference beats every learned agent" was scoped to "wherever the evidence is observable" on every
  surface.
- 30 summaries pulled back (sha256 match). Claims ledger: SEL1 block (45 numbers). SEL1 adapters and episodes
  released (task 1820: `actionproof-sel1-adapters` 1f13edf, traces dataset 87b5ae8).

## Phase D: demo and judge experience (10:15 IST)
- Live Space:
  - judge mode by default (scripted acknowledgement-trusting agent against the trained agent, live);
  - RECORDED MODEL RUN options from task 1790 (48 real runs; sha256 bd88218a);
  - an explicit, labelled fallback when a live run fails;
  - a JSON export of every run.
- Deployed to staging first (61b1c77), then live (cf90ad3; rollback target 3bb0dc8). Both were tested through the
  API (`docs/submission/DEMO_RUN_OF_SHOW.md` checklist).
- **Open (author):** the live Space has no `HF_TOKEN` secret, so live base-model runs fall back, labelled, to the
  recording. Two checks still need a browser: the layout and the cold start.

## Phase E: narrative surfaces (10:15 IST)
- SEL1 filled, from `sel1_summary.json` only, into:
  - the README, the write-up (abstract 499 words, §1a item 7, §22, §25, §27, §34) and the slides (§5, A4);
  - the deck (fails and a-sel1 slides; artifact version 1795754373-a72e);
  - `FINAL_STORY_DECISION.md` and a dated addendum to the novelty claim 3.
- Video script and captions rewritten to the judge-mode demo (2:45).
- Impact projection rerun with the SEL1 arms.
- Companion label added to `paper/main_v3.tex`; PDF rebuilt and committed.
- Release tables and write-up §28 now come from `release/ARTIFACT_MANIFEST.json`: 11 Hub entries public
  anonymously, 6 splits match, 13 of 13 result files present.
