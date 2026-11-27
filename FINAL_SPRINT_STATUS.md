# Final sprint status (RANK-1 directive)

Branch `final-sprint/winning-submission`, cut from `stage2-superiority-sprint` at 85657de.
Safety branch `final-sprint/backup-20261128-0200` (same commit). Local only; nothing pushed in this sprint yet.

Cutoffs: conservative freeze 2026-11-28 23:59 IST (binding until the portal says otherwise);
official 2026-11-28 23:59 AoE = 2026-11-29 17:29 IST.

## Checkpoint 1 — 2026-11-28 02:04 IST

**State at start.** `origin/submission-candidate` = 7ae863f (not yet fast-forwarded by the author);
`origin/stage2-superiority-sprint` = 85657de. Untracked build/paper artefacts and a pre-existing modification to
`slurm/tasks/1600_judge_next.sh` are left untouched.

**Compute.** sharanga worker `ap_w_h100b` (job 460650, 1 x H100 80GB) idle, about 22 h of wall time left; queue
empty. A D eval of txn_test (1,320 episodes) takes about 9.5 min.

**Largest judge-facing gaps (from the three hostile reviews).**
1. "Checks plus agent" was never measured; a hand-written script (B4) beats every trained agent.
2. Safe escalation is not solved (D: CER 15/540 over 3 seeds; Dw_aug 0/540).
3. The video is not recorded; human judge-calibration labels are missing (author actions).

**Workstreams started.**
- W2/W4 Recovery Gate (RG1), coordinator: evidence-only runtime controller around a model agent; fresh sealed split
  `rg1_test`; registration before any test output.
- W1 competitor refresh (agent): current HEADs of all 20 repos; the three required competitor files.
- W7 judge Q&A, pitches, REPRODUCE.md (agent).
- W6 impact calculator and figure set (agent).

**Next decision.** Freeze the gate after the dev pilot; register RG1; launch the test runs.
