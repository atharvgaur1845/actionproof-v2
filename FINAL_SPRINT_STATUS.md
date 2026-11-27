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

## Checkpoint 2 — 2026-11-28 02:30 IST

**Done.**
- **Recovery Gate (RG1) built:** `actionproof/recovery_gate/`, 23 tests, design spec
  `docs/research/RECOVERY_GATE_PROTOCOL.md`. Sealed split `rg1_test` generated at 02:11 IST (440 tasks, sha256
  0c38e439), never run.
- **Dev pilot 1** (task 1900, txn_dev, D seed 0, the first gate version):

  | | VTC | STR | FCR | DMR | CER |
  |---|---|---|---|---|---|
  | No gate | 285/360 | 285/360 | 18/360 | 13/360 | 0/60 |
  | Gate | 277/360 | 337/360 | 0/360 | 4/360 | 60/60 |

  The 8 lost completions all come from one flaw: after D refunded a partial payment, the gate bounced D's correct
  repay back to it. Fixed (read first, then decide), with concrete rejection messages and a stable-key rule; pilot 2
  (task 1901: txn_dev and the untouched txn_val) is queued.
- **Live Space** (not yet deployed): gate demo (no GPU) and trained agent behind the gate (live); 78 tests pass.
- **Forks:**
  - W1: competitor refresh. Only 4 of 20 repos changed; organisers asked for toolkit citations, now added after
    checking the arXiv IDs.
  - W6: impact calculator and figures 1, 3, 4, 6.
  - W7: judge Q&A, pitches, REPRODUCE.md.
  - W7b: a 9-page short paper of the main result.

**Running.** Task 1901 (pilot 2).

**Next decision.** If pilot 2 shows no completion loss, register RG1 and queue tasks 1910–1929 (about 3 h of H100
time).

## Checkpoint 3 — 2026-11-28 02:46 IST

**Registered, running on the H100 worker (36 tasks, about 3 h):**
- **RG1** (43e000c, 02:37): D seeds 0–2 with and without the gate on the sealed rg1_test, plus:
  - five rule ablations;
  - base, Dw_aug, SEL and Aya Expanse 8B with and without the gate;
  - scripted arms;
  - a txn_test replication.
- **RG1 addendum 1** (5c3aa08, 02:41, descriptive): the frozen gate on sel1_test's transient outages. Prediction,
  stated before the runs: over-escalation on t2.
- **RG2** (4b635d2, 02:45): gate v2 (a bounded re-read, SEL1's 3-read budget) against the RG1 gate on a fresh
  sealed rg2_test (240 tasks).

**Pilot 2** (txn_val, D seed 0, final gate; development data only):

| | VTC | STR | FCR | DMR | escalation (unresolvable) |
|---|---|---|---|---|---|
| No gate | 285/360 | 285/360 | 19/360 | 18/360 | 0/60 |
| Gate | 297/360 | 356/360 | 0/360 | 1/360 | 59/60 |

**Deployed.** Live Space 511583d: gate demo (11 s, no GPU) and the trained agent behind the gate (live), API-tested on
staging (4270efb) and live. Rollback target bac5889.

**Also done.**
- A hostile code review of the gate is running (independent agent).
- verify_numbers.sh now recomputes RG1, its addendum and RG2.
- The release lists add the RG episodes and D's warm-start rows.

**Next.** Integrate the results into README, write-up, paper, slides, deck, judge Q&A, pitch, impact page, figures,
claims ledger, replay site and Space. Then release, run the clean room, a final hostile review, and the freeze.
