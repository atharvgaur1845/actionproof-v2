# Next actions (updated 2026-11-11 09:30 IST)

The experiment freeze is 2026-11-13 18:00 IST. SPEC §26 is the original kickoff list and is kept as history. This file is the live list, and each item points to its evidence.

## GPU queue (no new runs proposed)

| Pool | State | Tasks |
|---|---|---|
| h200 | worker 439065 running until 13:27; 443167 pending (~12th) | M2b done; `396a` sampled honesty eval (running) → M1 s1 → M2 s1 → a5lo ack s0 → 400a confirmatory |
| h100 | worker 443755 blocked (gpunode5 reserved until 11-13 12:13) | (seed-1 tasks moved back to h200 at 12:05) |
| a100 | worker 443756 pending (~20 ahead) | M3′ `outcome_G` s0 → M5 `tool_trust` s0 |

**Checkpoint 12:00 IST.** If the h100 or a100 worker has not started, move its tasks back to whichever pool has a running GPU.

**Decision rules.** These apply when the results land; they are pre-registered in `PREREGISTRATION_A6.md`.
- **M2b vs M1 and M2.**
  - If M2b ≈ M1 on HFR (both ≈ 0), F_under is implicated in suppressing honest failure.
  - If M2b ≈ M2, ND itself is the cause.
- **Seed 1.** If the M1 − M2 sign flips, report P-B as unstable.

## CPU analyses (from `docs/research/novelty_scouts_2026-11-11.md`)

1. **Done:** evidence → report, retry structure, F4 status, credit-workflow integrity. Output in `results/a6/evidence_use.md`.
2. **Training-time reward-term audit (G1).**
   - Compute per-group ND / F_under variation and its advantage contribution for M1 and M2.
   - Source: the GRPO rollout logs, if components are recoverable; otherwise rescore stored verdicts.
   - This decides whether the "honesty-term paradox" can be stated.
3. **Discovery / exploitation table (G2).**
   - Rates of reads showing the write missing vs retries and honest reports, per tool, before vs after GRPO.
   - Splits: v2 (`results/eval/a4/mw`) and A6.
4. **Credit-workflow collapse follow-up.**
   - Is the refund re-issued right after a `credits: []` read? This tests the substitution hypothesis.
   - Does it already occur in v2 `pcstr_d` on `e_heldout`? Expected no: v2 called the credit tool once.
5. **Zero-VoI proposition (G5).** Write one paragraph for the article (§3 of the draft).
6. **Fetch the remaining cluster artifacts** into `results/` (A6 seed 1, M2b, M3′, M5 as they land).

## Submission (in progress)

- [x] Abstract rewritten around the selection framing with corrected numbers (`docs/ABSTRACT.md`).
- [x] Article draft corrected: pooled +19.25 pp, seed spread, the `llm_judge` arm gave no signal, reads transfer but recovery does not. New A6 section, with pending markers.
- [x] README headline and links updated.
- [ ] Demo rebuilt around "it read the ledger, saw nothing, and said done". Use an M1 refund-UR episode, the M2 honest-failure counterpart, and the credit-collapse episode.
- [ ] AuditKIT bridge run on the final files. If it is not run, do not claim the toolkit chain.
- [ ] Archive the superseded strategy blueprint under `docs/audit/`.
- [ ] Final numbers pass after the freeze: confirmatory H1 (task 400), seed 1, M2b, M3′, M5.
