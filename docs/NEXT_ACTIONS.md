# Next actions (updated 2026-11-11 21:45 IST)

**Deadline:** 2026-11-15 midnight IST. **Experiment freeze:** 2026-11-15 18:00 IST (set by the user 2026-11-11), leaving 6 h for the numbers pass.

The plan comes from the Fable 5.2 research review: Phase 0 → 3, registered as `PREREGISTRATION_A7.md`. SPEC §26 is the original kickoff list, kept as history.

## GPU queue (h200 pool; tasks run in name order and respect `# after:`)

**Workers**

| Job | State |
|---|---|
| 443167 | running until ~2026-11-12 18:20 |
| 444520 | pending (submitted 21:40) |
| h100 443755 | blocked by a node reservation until 11-13 12:13 |
| a100 443756 | pending; its tasks were moved to hold |

**Tasks**

| Order | Task | What | Est. |
|---|---|---|---|
| running | `396q` | M2 `pcstr_d_noreport` seed 1 (A6), then evals | ~2.7 h |
| 1 | `397a` | Forced-prefix battery on all A6 arms and checkpoints, both schemas (E7-1) | < 1.5 h |
| 2 | `400a` | Confirmatory H1, as registered | ~3 h |
| 3 | `410` | A7 warm start (A6 rows + 20 credit demos with their own keys) and gate G-A7 | ~2 h |
| 4 | `411` | A7 `pcstr_d_noreport` s0 (H7-1) | ~3 h |
| 5 | `412` | A7 `ack_outcome` s0 (H7-2) | ~3 h |
| 6 | `413` | M2c `pcstr_d_noFunder` s0 on the A6 init (H7-4). No dependency, so a second worker can take it in parallel | ~3 h |
| 7 | `414` | Battery on the A7 arms and M2c | < 1 h |

**On hold**
- `411b`: A7 state arm, seed 1. Release only if H7-1 holds at seed 0.
- M2b seed 1: release only if M2 seed 1 replicates.
- M5, M3′, a5lo and all A5 runs: they cannot change a conclusion.

## Decision rules (PREREGISTRATION_A7 §5)

| Trigger | Rule |
|---|---|
| M2 seed 1: sampled P(failed \| stop-mode) < 0.3 | Drop the honest-failure observation |
| Battery: M1 plain-schema P(failed) on prefix a > 0.5 | Replace "suppression" with "not robust to the schema shift" |
| H7-1: A7 state arm's non-colliding credit-call rate on `e_heldout` | < 20%: kill the template-inheritance thesis. ≥ 50%: supported |
| H7-2: ack arm moves > 10 pp from the A7 init | Selection is not the cause |
| H7-4: M2c vs M2 | Decides whether ND or F_under removes failure reports |

## Phase 0 (no GPU): status

- [x] Disclose D-ENV-1 (idempotency-key collision) with its impact table, and D-EVAL-1 (schema shift). SPEC §27, article, abstract.
- [x] Honest failure restated as a conditional on stop-mode.
- [x] M1 seed 1 = 0/150 recorded.
- [x] Zero-VoI proposition corrected (third assumption; drift, not extinction).
- [x] Template markers and functional credit added (`results/a6/evidence_use.md` §5).
- [x] Amendment 7 registered (`results/a7_hashes.json`).
- [ ] H7-3: per-episode template-marker accuracy (CPU, once A7 arms land).

## Submission (Phase 3)

- [ ] Final numbers pass after the freeze: confirmatory H1, M2 s1, battery, A7 arms, M2c.
- [ ] Demo: an init credit episode (a silently void credit call) vs a state-arm episode (a substituted refund) vs an A7 arm episode.
- [ ] AuditKIT bridge run on the final files. If it isn't run, do not claim the toolkit chain.
- [ ] Archive the superseded strategy blueprint under `docs/audit/`.
- [ ] Reposition the paper around reward-channel selection among SFT templates. The channel ablation is the verified backbone; cite 2607.03478 as the nearest prior work.
