# Study history: every registered study, from its registration and result file (addendum §1)

2026-11-25, branch `submission-candidate`.

**Sources for each row:**
- the registration (`PREREGISTRATION_*.md`);
- the freeze file `results/<id>_hashes.json`, giving the commit that added it, `git log --diff-filter=A`;
- the result file and key named in the row;
- deviations in `docs/stage2/DECISIONS.md` (D-codes);
- red-team notes from `RED_TEAM_VERIFICATION.md`.

Claim-level numbers are also checked by `scripts/s2_numcheck.py` (635 numbers, 0 failures at commit 7d06fe8) and re-read in `MASTER_EXPERIMENT_LEDGER.md` (624/624).

**Model:** Tiny Aya Global (3.35B) with LoRA r16 unless stated. **Optimizer:** TRL GRPO with one update per batch (`OPTIMIZER_AUDIT.md`). **Adapters** are in `runs/<id>` on the cluster, and in `results/run_manifest.json` (stale for P1/AD counts). Published adapters: `ADAPTIVE_RL_AUDIT.md` header.

## 1. Stage 1 (SupportSim, e-commerce; Nov 1–15), briefly

| Study | Frozen | Hypotheses → outcome | Result file | Notes |
|---|---|---|---|---|
| Stage 1 confirmatory (`PREREGISTRATION.md`) | 5d58308 (11-04) | H1 PCSTR-d > best baseline: **supported**, 0.8425 vs 0.8275, +1.5 pp | `results/confirmatory.json` | PCSTR-d is a reward. Strict-metric and stress amendments: A10 |
| A4–A7 (reward channel, ablations) | `a4`–`a7_hashes.json` | exploratory channel result (S1-CHANNEL); A5–A7 not carried into the ledger | `results/eval/*` | — |
| A8 + addenda (X1 intervention) | `a8*_hashes.json` | interaction +0.678 [0.605, 0.750], **supported** (one seed per cell, development split) | `results/a8/x1_interaction.md` | D-A8-1 (resume fix) |
| A9 (grounded judge) | `a9_hashes.json` | AuditKIT TaskCompletion with the ledger accepts 213/298 false completions (0.715) | `results/judge/aya32b_grounded__summary.json` | reproduced by the main session's judge audit |
| A10 + addendum (strict metric, v2 stress) | 71deb02 | v2 stress: PCSTR-d 0.176, outcome 0.118, tool_trust 0.706 (**ranking reverses**) | `results/a10/v2_env.md` | addendum not frozen |
| A11 (gateway estimands) | 2c83930 | C3 **not supported** (interaction rejects for seed 0, not seed 1). H7 keys alone 0, verify-before-retry all: supported | `results/s2/a11_summary.json`, `results/a11/h7_matrix.json` | — |

## 2. Stage 2 (ProcureSim)

| Study | Frozen | Split | Arms × seeds | Hypotheses → outcome (result file · key) | Deviations | Red-team / audit notes |
|---|---|---|---|---|---|---|
| **A12** + addenda 1–2 | 0cf91d9; a12b f1ad004; a12c d31c75c | proc_conf (300) | R_V, R_A × 5 | H2 phantom read-back R_V 1.00 vs R_A 0.00 in 5/5, VSC +0.27 to +0.32: **supported**. H3 sign agreement 0.6 vs ≥ 0.8: **not supported** → composite C1 **not made**. Gate 0 failed → addendum 1 gate (`results/s2/a12_summary.json`, `h3_test.json`, `results/a12/gate.json`) | D-A12-1 (stop rule fired for R_A s0: saturation); D-A12-2 (R_V s2 crash at step 199) | R_A is saturated (zero-variance 0.78–0.81) and equals the warm start on 500/500 tasks. "Observability" is not identified separately from incentive (R1.5) |
| A13 (contract coverage) | 4323c5c | finite fault set | — | full contract accepts none of 194 single-fault and 479 two-fault violations (`results/a13/h5_finite.json`) | D-A13-1 (Fisher test replaced: deterministic rows) | — |
| A14 (forced evidence) | 0d8a955 | forked states | warm, R_V, R_A × 3 | BCA 1.00 for every policy: **supported** (the original "interpretation deficit" thesis abandoned) (`results/a14/a14_summary.json`) | — | — |
| A15 (unseen ratios) + A16 (mechanism) | 440053a | proc_mech | R_V × 5, R_A × 3 | A15: repair at unseen ratios as often as trained in seeds 0–3; seed 4 0.0 (`results/a15/a15_summary.json`). A16: probe AUROC 1.00 everywhere; patching rule met 0/5; ablation 5/5 but not R_V-specific (`results/a16/a16_summary.json`) | D-A16-1 (CircuitKIT never ran) | `INTERPRETABILITY_AUDIT.md`: decodability only |
| A17 (OpsSim domains) | d231611 | care_admin, it_ops | R_V, R_A × 5 | R_V − R_A read-back +0.37 [0.18, 0.58] (criterion met); R_V vs warm +0.15 [−0.06, 0.39] **not established**; it_ops no difference (`results/a17/a17_seedavg.json`) | D-A17-1 (seed-averaged interval computed by a separate script) | negative transfer at a floor |
| K1 (pool composition) | b619220 | proc_conf | R_A targeted, placebo × 3 | phantom read-back 0/153 in all six (`results/eval/k1/conf`, `results/k1/k1_reward_signal.json`) | D-K1-1 (launch failures before training) | pool stays saturated (R1.5) |
| **S3** (RSC) | 5fd2fa4 (12:25 11-21) | s3_conf_rec, s3_conf_full (300 each) | RSC vs A12 R_V × 5 | H1 rec +0.206 [0.168, 0.247], 5/5, sign-flip 1/32: **supported**. H2 full non-inferiority: **supported** (`results/s3/s3_summary.json` `splits.*.across.vsc`). Recipe = start states + extra fault coverage (addendum 1 §1) | seeds 0–1 trained before registration with the frozen config, never evaluated before it (DECISIONS 11-21 12:30) | cell-clustered CI [0.060, 0.365] full, [0.098, 0.322] rec. WRITEUP §6 cites the post-hoc p = 0.004 as if registered |
| S3 addendum 1 | 1eb2bf8 | same | RSC vs class-matched control × 3 | H3 +0.09, H4 +0.22 "supported" by task CI with 3 seeds (`results/s3/s3_addendum.json`, `_full_pd.json`) | interval corrected [0.16→0.15, 0.29] (CONSISTENCY_AUDIT #20) | superseded by addendum 2 |
| **S3 addendum 2** (Part A, B) | 9bf99ca | s3_conf_*, s3_pp_* | RSC vs control × 5 (A); rebalanced RSC, control × 2 (B) | A: H3′ +0.107, Holm 0.036 **supported**; H4′p +0.384, Holm 0.111 **not**; H4′d +0.130, Holm 0.337 **not** (`results/s3/s3_addendum2.json` `part_a.tests`). B: receive-then-pay repair 0.005, reading "not explained by pool frequency" | — | learners 5/5 vs 2/5, Fisher 0.083. Full-episode cell CI now [0.014, 0.200] (`results/audit/redteam_recompute.json`) |
| S3 addendum 3 (STATE arm) | 198bea0 | — | — | **withdrawn before training**: degenerate design (`docs/research/ADDENDUM3_WITHDRAWN.md`) | — | red team §3.10 |
| **S3 addendum 4** (ACK starts) | 673056e | s3_conf_full partial | ACK × 5 vs RSC, control | D1 RSC − ACK +0.203, Holm 0.206; D2 ACK − control +0.181, Holm 0.262: **neither**, "inconclusive" (`results/s3/s3_addendum4.json` `primary`) | — | RSC − ACK on all tasks not cell-robust |
| **S4-EVAL** + addendum 1 | f12ce64; 50a9b3d | s4_api, s4_lang_*, s4_wf4, s4_combo | RSC vs A12 R_V × 5 | G1 API −0.016 **not**; G2 Hindi/Hinglish +0.127/+0.072, Holm 0.016 **supported**; G3 wf4 +0.016 **not**; G4 two-fault +0.072 **supported** (`results/s4/s4_summary.json` `family`) | D-S4E-1 (Holm rule unattainable, fixed before output); D-S4E-2 (frozen script stopped on a pairing bug) | "survives" = coverage gains; the language shift touches only the request |
| **ID1** | 125d0ee; d1 ae746bb | proc_conf | R_V from a no-read warm start × 2 | phantom read-back 0/153 both seeds; training read rate 0.00 every block → registered reading "**selection**" (`results/id1/id1_summary.json`) | D-ID1-1 (eval guard; re-run) | 0/164 sampled reads at T = 1 and 0.98 zero-variance groups: the null is mechanical; scope "not discovered in this budget" |
| **SR1** | 198bea0 | s4_api, s3_conf_full | SR × 3 vs RSC s0–2 | SR-H1 +0.00 **not**; SR-H2 −0.248 **not** (`results/s4/sr1_summary.json`) | — | — |
| **E1E2** | d07e92f | e1_pp_*, e2_pp (fresh) | 21 frozen models | E1 P1–P3 (sampled partial VSC contrasts): **none supported** (Holm 0.103, 0.095, 0.433). E2 H1 position: **not**; H2 reconciliation line +0.48 "computation helps" (`results/e1e2_summary.json`) | — | E2 H2 motivated P1 arm B, which then failed on test (−0.030) |
| FCA (dev, kill criterion) | `docs/research/algorithm/FCA_GRPO.md` §7 | s3_rec_dev | FCA × 2 | −0.097 vs RSC: **dropped** (`results/s4/fca_dev_summary.json`) | — | IS weight critique confirmed (`ADAPTIVE_RL_AUDIT.md`) |
| C5, G (dev, kill criteria) | `C5G_DEV.md` (`results/c5g_hashes.json`) | s3_rec_dev, proc_dev | × 2 each | −0.32/−0.37 and −0.11/−0.033: **dropped** (`results/s5/s5_dev_summary.json`) | G s1 re-run after an HF bundle failure | — |
| **R1** | 9b85f17; d1 95ffcac | IFEval 541, BFCL v4 11 categories | 35 models (5 seeds × 2 arms + warm + base …) | H2 non-inferiority within 0.03: **supported** both arms, both benchmarks; H1 SFT costs capability: **not**; H3 abstention: **not** (`results/r1/r1_summary.json`) | D-R1-1 (chat template) | retention, not improvement |
| **R1_TAU** | 6342585; d1, d2 | τ-bench retail test 115 × 3 | base, SFT, 2 GRPO arms | descriptive: base pass^1 0.058; SFT −0.020 [−0.038, −0.006]; GRPO ≈ 0 (`results/r1_tau/r1_tau_summary.json`) | D-R1T-1, D-R1T-2 | floor regime |
| **J1** + addendum 1 | 8af56da; a1 bd60beb | 660 main + 76 calibration | 4 judge families × views | J-H1 transcript judges fooled ≥ 0.25: **not** (≤ 1/120); J-H2 snapshot ≥ 0.90 Wilson LB for 3 of 4: **not** (0.886–0.989) (`results/j1/j1_summary.json`). Addendum 1: removing one rubric sentence turns abstention into acceptance (120/120, 120/120, 115/120, 9/120) (`results/j1/j1_addendum1.json`) | — | no human labels |
| **J2** | 3f5adad | same cases | Aya 32B + J1 four | descriptive. Aya transcript FAR 0.700, **with snapshot 0.573**, against ≤ 0.013 for the four J1 judges. Order and repeat agreement 0.92–0.99. Error-weighted J1 committee, transcript view: 0.839 accuracy at full coverage. Human: none (`results/j2/j2_summary.json`) | §6 of registration | **no ledger claim of its own** |
| **P1** | 31fc630 | p1_test_* (450, fresh) | A, B, C, D × 3 | H1 B −0.030 **not**; H2 C +0.259 Holm 0.0003 **supported**; H3 D +0.306 Holm 0.0003 **supported**; dev winner D (`results/p1/p1_summary.json` `test`) | D-P1-1 (AuditKIT sample fixed before test output) | C and D not nested; `p1_dwarm` never evaluated (`P1_DECOMPOSITION.md`) |
| **P2** | be01527 | same 450 | E, B2, Ag (+ Cg, Dg, Eg) × 3; refs W, M, O | H2 E − A +0.269 **supported**; H1 E − D −0.038, H3 B2 −0.260, H4 Ag −0.122, S1–S3 **not** (`results/p2/p2_summary.json` `primary`, `secondary`) | — | guard rule R1 blocks the registered repair (`results/quality/p2_guard_diagnosis.json`) |
| **LAB1** | 1bbbdea | tabular testbeds | CDVO family vs SFB × 5 | L1 H1/H2 **not** (0/5 seeds); L2 H4 normaliser intervention **supported**, H3 **not** (`results/lab/lab1_summary.json`; claims LAB-*) | §7 | — |
| **TH1** | 5f1100e | 731 held-out analytical problems | GRPO, no-std, RLOO, … | H1–H5 **supported**; H6 **not** (53/75); H7 **untestable** (3 pairs) (`results/theory/th1_heldout_summary.json`; paper v3 §TH1) | §6 | — |
| **AD-C1** + addendum 1, notes 2–3 | 99159bb; a1 494ba06 | cw_conf (300) | GRPO, no-std, AWRB, AWRB-no-std × 5 | H_nostd +0.120 Holm 0.016 **supported**; H_awrb +0.192 **supported**; H_awrbn +0.032 Holm 0.048 **supported**; H_AWRB −0.088 **not** (predicted); H_D3 **not run** (note 3) (`results/ad/c1_summary.json` `results`) | note 2 (arm entry rule), note 3 (D3 seeds 2–4 not queued) | GRPO 0.687 in all 5 seeds: greedy degeneracy |
| **P3** | 7074e8d (11-25) | p1_test_* | Dn × 3 vs D | running (main session) | — | — |

## 3. Where documents differ from result files (flags for the main session)

Numbers: none differ (number check 0 failures; ledger 624/624). The differences are in wording and statistic labels:

1. **WRITEUP §6, row S3**: "seed permutation p = 0.004" is printed as the support statistic. The registered rule was ≥ 4/5 seeds plus an interval excluding 0 (sign-flip 1/32). The p = 0.004 is the post-hoc S3-R statistic, and should be labelled so.
2. **README l. 57, WRITEUP §21 item 8**: C (0.772) and D (0.819) are shown side by side. C is not D's warm start, so the gap is not GRPO's contribution (`P1_DECOMPOSITION.md`).
3. **README l. 15, WRITEUP l. 23 and 51 (ID1)**: "does not create it" matches the registered reading. Its scope should state that no read was ever sampled (0/164), so the reward never saw one.
4. **WRITEUP §21 item 1, "Reward observability selects verification"**: the red-team identification objection (R1.5) is unaddressed; the narrowed wording was not adopted.
5. **J2**: no ledger claim. Aya's 0.573 false acceptance with the snapshot is unclaimed.
6. **paper v1 (`paper/main.tex` l. 109, `paper/appendix.tex` l. 79)**: the FCA all-action derivation is still wrong (superseded by v3; matters only if v1 ships).
7. **`docs/research/FINAL_REPORT.md` l. 52–53**: arm means are worded as per-episode ranges (historical Stage 4 report).
8. **Optimizer naming**: fixed in this audit (commit 7d06fe8).
