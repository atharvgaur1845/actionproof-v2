> **Superseded (2026-11-27, Stage 2 sprint).** This file is kept as history (last updated 2026-11-25) and is not current. The current source is `claims/claims.yaml`, checked by `scripts/s2_numcheck.py`. See `research_audit/STALE_SURFACES_AUDIT.md`.

# Claim verification

2026-11-25, branch `submission-candidate`.

- **Raw data:** 95,100 stored episodes in 242 result files were re-read.
- **Script:** `scripts/audit/claim_verify.py`, run on Slurm (jobs 469855, then 469858 with the second pathway).
- **Output:** `results/audit/claim_verify.json`.

No historical result file was changed.

## 1. Method: two pathways separate from the original code

The reference verifier `scripts/reference_verifier_proc.py` was written from `docs/stage2/VERIFIER_SPEC.md` alone and imports no simulator, verifier or reward code.

| Pathway | What it trusts | What it checks |
|---|---|---|
| **Replay** | Each recorded attempt's `committed` field (the environment's commit bookkeeping) and the task's initial state | Rebuilds the database and recomputes every verdict |
| **Final state** | The recorded final database | Recomputes every verdict with the reference logic |

**Aggregates are recomputed with plain means:** task means first (samples averaged within a task), then the mean over seeds. This is the registered statistic, implemented separately. Bootstrap intervals and permutation p-values are not recomputed; only point estimates and per-seed values are.

**Dependence on commit bookkeeping.** Replay depends on the environment's `committed` labels. The final-state pathway depends on the simulator's stored rows. Neither is independent of the simulator itself; an error in the simulator's fault injection would pass both. The golden-trace suite (91 hand-built cases) covers that layer, separately.

## 2. Integrity checks (all 242 files)

| Check | Result |
|---|---|
| Duplicate (task, sample) pairs | **0** |
| Tasks missing from a file, against its frozen split file | **0** |
| Task ids not in the frozen split | **0** |
| Final-state pathway agrees with the recorded VSC, per episode | **100.0% in every file** (A12, S3, P1, P2) |
| Replay pathway agrees, per episode | A12 100%. S3, P1, P2 96–99% (minimum 95.8% per file) |

**Why replay disagrees on S3, P1 and P2.** Every disagreement found is a partial write; on the inspected file they are all recovery-start episodes. The reference spec (proc1) assumes a partial commit stores exactly half. Stage 3 and P1 use other fractions, and P1's test deliberately uses unseen ones.
- Example: on task `p1_test_rec-0017` replay rebuilds 11 received units where the stored database holds 16.
- The reference reports these as `state_match: false` (P1: 6,906 of 32,400 episodes; P2: 10,433).

This is a limitation of the reference specification, not of the recorded results. The reference verdict logic applied to the stored state reproduces the recorded verdict in every episode.

## 3. Headline claims

| Claim (where published) | Recorded | Recomputed here | Status |
|---|---|---|---|
| A12: phantom read-back, state reward 1.00 and acknowledgement reward 0.00, in all 5 seeds | 1.00 / 0.00 | R_V 248/248, 253/253, 253/253, 255/255, 250/250; R_A 0/153 ×5 (store-excluding variant) | **PASS**, see note a |
| A12: VSC, state − acknowledgement reward, +0.27 to +0.32 per seed | +0.27 to +0.32 | +0.313, +0.323, +0.313, +0.277, +0.273; replay and final state agree 100% | **PASS** |
| S3: recovery-start recipe vs state reward, +0.21 on recovery starts and full episodes | +0.21 | +0.206 (recovery), +0.209 (full); final state | **PASS** |
| S3 addendum 2: start-state effect vs class-matched control, +0.107 | +0.107 | +0.107 (0.8213 − 0.7140) | **PASS** (point estimate; p not recomputed) |
| P1: A 0.513, B 0.483, C 0.772, D 0.819 | as stated | 0.5128, 0.4830, 0.7719, 0.8193 | **PASS** |
| P1: deltas +0.259 (C), +0.306 (D), −0.030 (B) | as stated | +0.259, +0.307, −0.030 | **PASS** |
| P2: E 0.782, B2 0.253, Ag 0.391, Cg 0.527, Dg 0.547, Eg 0.640 | as stated | 0.7817, 0.2528, 0.3911, 0.5274, 0.5467, 0.6396 | **PASS** |
| AD-C1: GRPO 0.687 ×5; no-std 0.807; AWRB 0.879; AWRB-no-std 0.719 | as stated | 0.6867, 0.8067, 0.8787, 0.7187 | **PASS for aggregation.** Verdicts **UNVERIFIED** by a separate pathway, see note b |
| ID1: read-back 0/153 in both seeds | as stated | not recomputed | UNVERIFIED |
| R1: BFCL and IFEval retention | as stated | not recomputed (external harnesses) | UNVERIFIED |
| J1 and J2 judge accuracies | as stated | not recomputed here; the judge study (`JUDGE_RELIABILITY.md`) re-measures judges on new cases | UNVERIFIED |
| TH1 (registered simulation) | as stated | not recomputed; the registration itself uses an independent trainer | UNVERIFIED |
| Stage 1 confirmatory PCSTR-d 0.8425 | as stated | not recomputed (Stage 1 environment) | UNVERIFIED |

**No headline claim failed.**

**Notes.**
- **a.** The reference's literal read-back reading also counts idempotency-store replies (status ok, nothing committed) as phantom writes. That gives R_V 0.988–0.996. The original verifier excludes them, which matches the reference's documented interpretation A22. Published wording should say "after every phantom write that reached the database". Alternatively, cite the ratio counts above, which match exactly.
- **b.** AD-C1 runs in its own world (`actionproof/ad/worlds.py`) with its own verifier. The reference verifier does not cover it.

## 4. Statistical methodology

- **Task bootstrap.** Every task-bootstrap interval holds the trained models fixed (the "runs fixed" interval). It reflects task sampling only, not training-seed variability. Seed variability is handled by the seed rule (all seeds positive) or by seed-level permutation tests, never by pooling seeds as if they were tasks.
  - *Pseudoreplication check:* P1 and P2 bootstrap the seed-averaged task score, so samples and seeds are not treated as independent tasks. **Not pseudoreplicated.**
- **AD-C1.** The five GRPO seeds are identical (0.687), so the seed permutation test sees no baseline variance. The 252-assignment exact test is still valid, but the effect's uncertainty is mostly on the candidate side. Its minimum attainable p (1/252) is respected.
- **Multiplicity.** Holm within each registered family (P1: 3; P2: 4 + 3; AD-C1: 4; R1, J1, E1 as registered). Families are not corrected across studies. Each registration's hypotheses form a separate family, stated in advance.
- **Sidedness.** One-sided tests are used where the registration fixed a direction. A negative estimate gets p = 1, as for B, B2 and the guard.
- **Rounding.** Every published number is checked against its file at the published precision by `scripts/s2_numcheck.py` (628 numbers in 10 documents, 0 failures). The canonical file reports 0 mismatches.
- **Checkpoint selection.** Every registered study evaluates the final checkpoint fixed in its registration. No confirmatory output was used to choose a checkpoint; P1's development winner labels the demo agent only.
- **Repeated evaluation.** The test-like phases of `p2_eval.py` refuse to overwrite. The frozen tests were run once per model. P2's evaluator was shown equivalent to P1's (90/90 greedy, 360/360 sampled) before reuse.

## 5. Corrections logged

- `scripts/diagnose_failures.py`: the row labelled "VSC" was the goal-met rate. True VSC was added (commit after b359ff5). The descriptive tables were regenerated; no registered statistic was affected.
- Narrowed theory claims (Phase II): "494/495" counts runs, and the p·s > 1/2 rule needs state-level repairability.
