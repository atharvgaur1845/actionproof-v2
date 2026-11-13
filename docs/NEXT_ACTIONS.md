# Next actions (updated 2026-11-13 06:45 IST)

**Deadline:** 2026-11-15 midnight IST. **Experiment freeze:** 2026-11-15 18:00 IST (set by the user 2026-11-11), leaving 6 h for the numbers pass.

The plan comes from the Fable 5.2 research review: Phase 0 → 3, registered as `PREREGISTRATION_A7.md`. SPEC §26 is the original kickoff list, kept as history.

## GPU queue (h200 pool; tasks run in name order and respect `# after:`)

**Workers**

| Job | State |
|---|---|
| 443167 | running until ~2026-11-12 18:20 |
| 444520 | cancelled 03:35 by another user of the shared account; not resubmitted (remaining queue fits in 443167) |
| h100 443755 | blocked by a node reservation until 11-13 12:13 |
| a100 443756 | pending; its tasks were moved to hold |

**Tasks**

| Order | Task | What | Est. |
|---|---|---|---|
| done | `396q` | M2 s1: greedy refund-UR 6/50 failed (s0 45/50) | — |
| done | `397a` | Battery: plain-schema P(failed) ≥ 0.989 for every model → rule triggered ("not robust to the schema shift") | — |
| done | `400a` | Confirmatory H1 **passes**: +1.5 pp [0.63, 2.62], p = 0.025 | — |
| done | `410` | Gate G-A7 **passes**: HFR 0.21; non-colliding credit 200/200 | — |
| done | `411` | A7 state arm: **H7-1 supported**, non-colliding credit 200/200 (A6: 0/200) | — |
| done | `411s` | M2 s1 sampled 0.264 < 0.3 → **honest-failure observation dropped** (M1 s1 0.236) | — |
| done | `412` | A7 ack arm: **H7-2 holds** (credit 200/200); channel effect replicates (VSC 0.615 vs 0.845) | — |
| done | `413` | M2c `pcstr_d_noFunder` s0: H7-4 **inconclusive** (P(failed \| stop-mode) 0.42) | — |
| done | `414` | Battery on the A7 arms and M2c | — |
| pending | `415` | E7-5: A7 state arm seed 1 | queued on two pools with a run-once guard: h200 worker 445667 (~5 h) and a100 worker 446674 (`# minutes: 600`), both pending on priority at 11-13 07:00 |

**On hold**
- M2b seed 1: condition not met (M2 s1 did not replicate); stays held.
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
- [x] H7-3: template-marker accuracy 0.991 (1586/1600) ≥ 0.9, holds.

## Submission (Phase 3)

- [ ] Final numbers pass after the freeze: confirmatory H1, M2 s1, battery, A7 arms, M2c.
- [x] Demo tab "Same task, three templates" (`scripts/build_credit_demo.py` → `demo/replays/credit_templates.json`; 6 fault-free `e_heldout` tasks: void credit / substituted refund / committed credit).
- [x] AuditKIT bridge on the final files: 36/36 match VSC exactly (`results/auditkit/bridge_final.md`). The toolkit chain claim stands.
- [x] Superseded strategy blueprint archived: `docs/audit/strategy_blueprint_2026-11-10_superseded.md`.
- [ ] Reposition the paper around reward-channel selection among SFT templates. The channel ablation is the verified backbone; cite 2607.03478 as the nearest prior work.
