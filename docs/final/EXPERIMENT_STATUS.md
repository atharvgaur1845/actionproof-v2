# Experiment status

As of 2026-11-23 11:47 IST (branch actionproof-research-upgrade).

**Compute:**
- Slurm lease 460633: 2 H200 GPUs until Nov 25 10:27.
- A100 worker 460629.
- A 200-step GRPO run takes about 1 h 55 min on an H200 (`logs/*_s*.full.log`) and about 2 h 30 min on the A100.

**Submission:** Nov 27, 23:59 IST (moved by the author on 2026-11-23).

**Labels:**
- **Completed and verified:** the result file exists and is in the claims ledger.
- **Running:** registered, with jobs in progress.
- **Withdrawn / failed:** stated as such.

| Experiment | Registration | Seeds | Status | Result file | Expected |
|---|---|---|---|---|---|
| A12 reward observability | `PREREGISTRATION_A12*.md` | 5 + 5 | Completed and verified | `results/s2/a12_summary.json` | — |
| A14/A15 forced evidence | `PREREGISTRATION_A14.md`, `_A15.md` | 5 per arm | Completed and verified | `results/a15/`, `results/s2/partial_by_workflow.json` | — |
| S3 recovery-start vs state reward | `PREREGISTRATION_S3.md` | 5 + 5 | Completed and verified | `results/s3/s3_summary.json` | — |
| S3 addendum 1: vs class-matched control | `PREREGISTRATION_S3_ADDENDUM1.md` | 3 + 3 | Completed and verified | `results/s3/s3_addendum*.json` | — |
| S4-EVAL shifts | `PREREGISTRATION_S4_EVAL.md` + addendum 1 | 5 + 5 (+3 control) | Completed and verified | `results/s4/s4_summary.json` | — |
| FCA (forked counterfactual credit) | `docs/research/algorithm/FCA_GRPO.md` §7 | 2 (development) | **Failed** its kill criterion; dropped | `results/s4/fca_dev_summary.json` | — |
| S3 addendum 2, Part A: control seeds 3–4 (prospective) | `PREREGISTRATION_S3_ADDENDUM2.md` | +2 control | **Completed and verified.** H3′ supported (p 0.012, Holm 0.036); H4′p and H4′d not supported; safety trade-off reversed | `results/s3/s3_addendum2.json` | — |
| S3 addendum 2, Part B: workflow-rebalanced pools | same | 2 + 2 | **Completed and verified.** Reading "not frequency" (W = 0.005) | `results/s3/s3_addendum2.json` | — |
| S3 addendum 3: state-only starts | `PREREGISTRATION_S3_ADDENDUM3.md` | — | **Withdrawn before training** (degenerate design) | `docs/research/ADDENDUM3_WITHDRAWN.md` | — |
| S3 addendum 4: acknowledgement-only starts (is it the handed evidence?) | `PREREGISTRATION_S3_ADDENDUM4.md` | 5 | **Completed and verified.** Reading "inconclusive": RSC − ACK +0.203 [0.169, 0.238], Holm 0.206; ACK − control +0.181 [0.137, 0.225], Holm 0.262. Runs that learn partial repair: control 2/5, ACK 4/5, RSC 5/5 | `results/s3/s3_addendum4.json` | — |
| ID1: warm start that never reads, then R_V | `PREREGISTRATION_ID1.md` (deviation D-ID1-1) | 2 | **Completed and verified.** Reading "selection": 0/153 read-back in both seeds, no read sampled in training | `results/id1/id1_summary.json` | — |
| SR1: tool-name randomisation (a known fix, tested here) | `PREREGISTRATION_SR1.md` | 3 | **Completed and verified. Not supported.** Renamed API +0.00 [−0.04, 0.04]; on the canonical tools −0.248 [−0.296, −0.201] (non-inferiority fails) | `results/s4/sr1_summary.json` | — |
| E1/E2: sampled outcomes and decision-point probes (21 frozen models, fresh splits) | `PREREGISTRATION_E1E2.md` | 5 per arm | **Completed and verified.** Sampled RSC − control +0.098 (Holm 0.103), not supported; threshold reading (ρ 0.92); position H1 not supported as registered; reconciliation line +0.48 ("computation helps") | `results/e1e2_summary.json`, `docs/research/E1E2_RESULT.md` | — |
| Stage 5 development: C5 verified-progress turn credit, G hint-guided sampling | `docs/research/algorithm/C5G_DEV.md` | 2 + 2 | **Both dropped** by pre-set kill criteria (C5 −0.32/−0.37 with diagnosis; G −0.11/−0.033) | `results/s5/s5_dev_summary.json` | — |
| R1 capability retention: IFEval + BFCL v4 single-turn, 30 procurement models + Stage 1 chain | `PREREGISTRATION_R1.md` (deviation D-R1-1, before output) | 5 per arm (SR1 3) | **Completed and verified.** H2 supported (non-inferior); H1 and H3 not supported | `results/r1/r1_summary.json` | — |
| R1-T τ-bench retail (descriptive), with a feasibility pilot and an overflow cancellation rule | `PREREGISTRATION_R1_TAU.md` | 3 per arm | **Completed** (descriptive; floor regime; deviations D-R1T-1/2 before any episode) | `results/r1_tau/r1_tau_summary.json` | — |
| J1 judge committee: 4 families × 736 cases × 2 views + prompt variant; human calibration | `PREREGISTRATION_J1.md` | — | **Completed** (J-H1, J-H2 not supported; addendum 1 exploratory; $3.20); human labels pending | `results/j1/j1_summary.json`, `results/j1/j1_addendum1.json` | — |
| AD-D1/D2/D3 development (costed-verification testbed cw; GRPO, no-std, AWRB, AWRB-no-std; EMA scale; per-read cost) | `docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL*.md` | 2 per arm | D1 seed 0 and part of seed 1 done (GRPO collapses verification 2/2; AWRB passes gates); D1 seed 1, cost-0.30 interventions, D2, D3 **running** | `results/ad/dev/`, `results/ad/d*_summary.json` | D3 about 00:30 Nov 24 |
| AD-C1 confirmatory (GRPO vs AWRB, seeds 0–4, cw_conf once, Holm) | `PREREGISTRATION_AD_C1.md` + addendum 1 | 5 + 5 | **Queued / running overnight** | `results/ad/c1_summary.json` (`scripts/ad/c1_stats.py`) | about 06:00 Nov 24 |
| Mechanistic causal test (CircuitKIT) | — | — | **Not run.** CircuitKIT failed to install in Stage 2 (D-A16-1); A16's ablation is not specific | — | Future work |

**Not attempted, by decision:**
- the own-history recovery-start arm (needs new two-phase rollout code that is risky this late);
- a second model family;
- prompted or larger-model baselines on ProcureSim;
- sampled (T > 0) evaluation.

All are listed as future work in `docs/final/FINAL_REVIEW.md`.

**Rule for the final days:** results are reported as they land, whatever the sign. Nothing is stopped early or extended because of an interim number.
