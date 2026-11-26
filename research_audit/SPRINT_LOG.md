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
