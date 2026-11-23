# State of research (reconstructed 2026-11-24, branch `research/lab-final`)

Every statement here was checked against code or result files in this session, unless it is marked otherwise.
Evidence levels are kept separate; a claim appears under its strongest *verified* level only.

## 0. Repository and branch state (verified 01:30–02:10 IST)

| Item | Finding |
|---|---|
| Public remote `origin` (atharvgaur1845/actionproof-v2) | `main` = d19c299, `research/e1e2`, `stage4` = 6147ede. **Nothing from the upgrade or the v2 algorithm work is public.** |
| `actionproof-research-upgrade` (295a6a8) | local only |
| `research/algorithm-discovery-v2` (9738554) | local only; **22 commits** ae6456e..9738554, verified one by one (115e524 … 9738554) |
| Merge into the upgrade line | **9e5dd04** (parents ae6456e, 9738554) |
| `quality-upgrade` (508d2af) | local only; = upgrade + merge + P2 registration (be01527) + unified run manifest (508d2af) |
| This work | `research/lab-final`, branched from 508d2af; local only |
| v2 tests | `tests/ad2/`: **20 passed** (3.6 s) |
| v2 reports | 16 markdown files in `docs/research/algorithm_discovery_v2/`. The 13 result reports are: ABLATION, ALGORITHM_CANDIDATES, AWRB_AUDIT, CONTROLLED_TESTBED, FAILURE_ANALYSIS, FINAL_ALGORITHM_REPORT, GENERALISATION, GRPO_FAILURE_DERIVATION, LITERATURE, SELECTIVE_VERIFICATION, STATISTICAL_ANALYSIS, V2E1_CONDITION_B, V2TB7. The other three are protocol and logs: V2TB_PROTOCOL, DEVIATIONS, SMOKE_EXPERIMENT_LOG |
| v2 raw testbed outputs | `results/ad2/tabular/TB{1,2,3,4,6,7}.jsonl.gz` (TB1 11,664 runs) with summaries; frozen by `results/ad2/hashes_V2TB_PROTOCOL.json` |
| AD-C1 run configuration | Tasks 1220–1231 (training grpo / awrb / nostd / awrbn, seeds 2–4) and 1240/1241 (evaluation) are **pending** in queue/{h200,h100}/pending. They sort behind P1 (1155/1156 running, 1163–1172 pending) and P2 (1180–1185 pending; 1182 running on the H100). Their prediction is recorded: H_AWRB not supported |
| Tasks 1301–1303 | in `queue/a100/hold` |
| Task 1300 | in `queue/a100/failed`. **Cause verified:** `slurm/ad2_arm.sh` line 4 assigns `GROUPS=$3`. `GROUPS` is a read-only bash array; the assignment is silently ignored and `$GROUPS` expands to the user's primary GID. Log line: `unknown --groups 1248200074`. Not re-queued (§6) |

## 1. Mathematically derived

| # | Claim | Where | Check |
|---|---|---|---|
| M1 | **Theorem 1 (stake-free vote).** Under a hidden state shared within the group and deterministic within-state returns, GRPO's expected update on a binary decision is (G − 1)·m_G(ρ)·Σ_z π(z)·sign(β_z), while Dr. GRPO's is (G − 1)·ρ(1 − ρ)·VOI and RLOO's is G·ρ(1 − ρ)·VOI, exactly, for finite G | `docs/research_lab/GRPO_MECHANISM_FORMAL.md` §2 | exact enumeration, 21 tests (`tests/lab/test_exact_vote.py`) |
| M2 | Hence the vote rule: GRPO verifies iff p·s > 1/2, independent of c, V and L_f. Exact threshold 0.5000 for every c tested at ε → 0 | same | test |
| M3 | Normaliser condition: a scale independent of the samples' hidden states, given the context, preserves every decision's expected sign | v2 GRPO_FAILURE_DERIVATION §4 | proof sketch; LAB1 H4 tests it causally |
| M4 | Edge cases: ε, stochastic recovery, exogenous noise, condition B, c = 0 | FORMAL §3 | exact thresholds |
| M5 | The paired fork contrast (CDVO) is unbiased for Q^π(h, v) − Q^π(h, c) under conditions (i)–(iv); CRN cannot remove Var_z(Δ); drift bias of stored fork means is bounded | `docs/research_lab/CDVO_THEORY.md` | unbiasedness test with and without CRN |

## 2. Reproduced in the synthetic testbed (exact optimum; development, registered before output)

| # | Claim | Value | File |
|---|---|---|---|
| T1 | GRPO wrong-decision rate | **319/495 = 0.644** (independent recount) | `results/lab/exact_analysis.json` |
| T2 | Dr. GRPO wrong-decision rate | **28/495 = 0.057**; all 28 are VOI > 0 under-verification, 27 with \|VOI\| ≤ 0.05 | same |
| T3 | The exact warm-start sign predicts GRPO's learned decision | **494/495** cells; condition B 0.976 | same |
| T4 | GRPO under condition B still biased | 156/495 = 0.315 (v2 H1c failed) | same; v2 V-F1 |
| T5 | GDPO worst | 370/495 = 0.747 | same |
| T6 | igc selectivity, about 4× sample efficiency, same endpoint at 8× | TB2 / TB7 (igc_v1 0.0202 vs nostd 0.0296 at ×1) | v2 summaries |
| T7 | Forks fail at matched budget | TB2 (v2 V-F6) | v2 |
| T8 | LAB1 results | §8 below | `results/lab/lab1_summary.json` |

## 3. Observed in trained language agents (Tiny Aya Global; descriptive unless stated)

| # | Claim | Level | File |
|---|---|---|---|
| L1 | Training-log decomposition: −143.6 (std) vs −0.36 (centring); faulted groups +49.6 / −49.6 | **reproduced byte-identically** from raw rollouts on 2026-11-24 | `results/ad2/decomposition.json`; `results/ad/rollouts/ad_grpo_s0.jsonl.gz` (sha256 bb0c7a9c…) |
| L2 | AD-D1: GRPO collapses verification (2/2 dev seeds); nostd / awrb / awrbn pass G1–G4 | development | `results/ad/d1_summary.json` |
| L3 | AD-D2: EMA-scaled arms fail G1; one seed of each collapses to 0 reads | development, negative | `results/ad/d2_summary.json` |
| L4 | AWRB's development VSC gain is over-verification (journal verify 0.91) | development, negative | v2 AWRB audit |

## 4. Confirmatory (registered before output, tested once)

| # | Claim | Result | File |
|---|---|---|---|
| C1 | State reward → phantom read-back 1.00 (5/5) vs acknowledgement reward 0.00 (5/5) | supported as consistency; Holm 0.0625 | `results/s2/a12_summary.json` |
| C2 | Recovery-start GRPO (RSC) vs state reward: VSC +0.21 [0.17, 0.25], 5/5 | supported | `results/s3/s3_summary.json` |
| C3 | RSC vs class-matched control, 5 + 5: recovery starts +0.107, Holm 0.036 | supported; partial-specific gain not supported | `results/s3/s3_addendum2.json` |
| C4 | Shifts: Hindi, Hinglish, two faults hold; renamed API and new workflow fail | mixed | `results/s4/s4_summary.json` |
| C5 | SR1 (tool-name randomisation) | not supported | `results/s4/sr1_summary.json` |
| C6 | E1 sampled RSC − control +0.098, Holm 0.103 | not supported | `results/e1e2_summary.json` |
| C7 | J1: J-H1 (transcript judges fooled) and J-H2 (snapshot judges ≥ 0.90) | **neither supported**. Gemma is the most accurate judge (0.989 with the snapshot) | `results/j1/j1_summary.json` |
| C8 | R1 capability retention: H2 non-inferior supported; H1 and H3 not supported | | `results/r1/r1_summary.json` |
| C9 | AD-C1 (H_AWRB etc.) | **PENDING**: tasks 1220–1241 not run | — |
| C10 | P1 (practical agent, arms A–D, frozen test) | **PENDING** (lead engineer's run; test outputs not read here) | — |
| C11 | P2 | **PENDING** (registered be01527) | — |
| C12 | LAB1 H1–H5 (testbed, not LLM) | §8 | `results/lab/lab1_summary.json` |

## 5. Exploratory

- J1 addendum 1: v0 prompt (warning removed) fools every judge (Gemma 120/120). `results/j1/j1_addendum1.json`.
- E2: reconciliation line +0.48. ID1: reward selects, does not create, reading.
- P1 development: arm C (recovery SFT) highest dev VSC, 0.842 / 0.861 (`docs/research/P1_DEV_DIAGNOSIS.md`;
  development only).
- J2 (Aya baseline, robustness, aggregation): §8.

## 6. Negative (preserved)

- FCA dropped (−0.10 dev VSC). C5 and G dropped. S3 addendum 3 withdrawn.
- AWRB is not better.
- Forks fail at matched budget (v2).
- ctxema is not better than nostd.
- D2 EMA collapse.
- igc's TB2 gain is about one third an artefact (V-F9).
- H1c failed. H1e failed narrowly. SR1 failed. Negative transfer to healthcare administration.
- **V2-E1 (LLM condition B) never ran** (task 1300, GROUPS bug). Re-queuing would cost about 4 × 5.5 A100-hours
  behind P1/P2/AD. It would test only the weakened prediction "less collapse, not none". Not re-queued by this lab;
  the author decides (§9).

## 7. Unverified

- Calibrate-Then-Act's static-policy collapse as an instance of this mechanism (their group construction is unread).
- Any human-judge agreement: the 76 labels do not exist yet, and there is no second annotator.
- Gemma 4's licence: J1's registration lists "Gemma licence (model card)", but the Hub card metadata reads
  apache-2.0 (`results/j2_model_revisions.jsonl`). Check before the write-up.
- Whether any 2026 paper states Theorem 1's mixture result (searches were negative; weak evidence).

## 8. Results of this lab (filled from result files)

| ID | Level | Result | File |
|---|---|---|---|
| LAB1 H1 | testbed, registered (development) | **not supported**: adaptive forks 0.0170 vs Dr. GRPO 0.0080 at E4, 0/5 seeds | `results/lab/lab1_summary.json` |
| LAB1 H2 | testbed, registered | **not supported**: forks + igc 0.0097 vs igc 0.0062, 0/5 | same |
| LAB1 H3 | testbed, registered | **not supported**: CRN variance ratios 0.94–1.12 | `results/lab/grad_snr.json` |
| LAB1 H4 | testbed, registered | **supported**: same-state scale wrong 0.667, fresh-state scale 0.061 | `lab1_summary.json` (L2N) |
| LAB1 H5/T4 | testbed | fail (PPO generalises best) | L3G |
| L2G | testbed, exact | GRPO's mean gradient is wrong-signed on every risky verify logit (cos −0.175) | `grad_snr.json` |
| LLM read rates | trained agents, development | grpo read share 0.19 / 0.20 → 0.02 / 0.04; Dr. GRPO 0.23 / 0.28 → 0.48 / 0.50 | `results/lab/train_read_rate.json` |
| J2 | exploratory (registered) | see `docs/research_lab/JUDGE_RESULTS.md` | `results/j2/j2_summary.json` |

**CDVO verdict: rejected** (fails novelty, H1, H2, generalisation). No GPU task was registered for it.

## 9. Decisions only the author can make

Listed in `docs/research_lab/FINAL_REPORT.md`.
