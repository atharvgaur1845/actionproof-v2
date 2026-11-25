# Adaptive and algorithmic RL audit (addendum §2C)

2026-11-25, branch `submission-candidate`. Every training method in the repository that changes something other than the reward's information source.

Each status comes from:
- its result file;
- its decision rule, fixed in advance.

Run existence is checked against `runs/` on the cluster (`grpo_training_config.yaml` present). The optimizer underneath every row is the same TRL 1.7.1 update (`OPTIMIZER_AUDIT.md`), unless the row says otherwise.

**Publication.** Adapters published to the Hub are those listed in `scripts/publish_procure.py`:
- the warm starts;
- A12 R_V/R_A;
- RSC; the class-matched control; the rebalanced RSC and control;
- ID1;
- FCA.

Every other adapter exists only on the cluster (`runs/`). That includes P1, P2, AD-C1, ACK, SR1, C5 and G.

## 1. The table

| Method | Code | What it changes | Trained? | Judged against | Outcome | Confirmatory result | Reproducible from repo + checkpoints | Status |
|---|---|---|---|---|---|---|---|---|
| PCSTR-d (Stage 1 reward) | `actionproof/rewards/`, `actionproof/train/grpo.py` | reward: process-conditioned state truth | yes (Stage 1, SupportSim) | `PREREGISTRATION.md` H1 vs best baseline | supported, +1.5 pp (0.8425 vs 0.8275) | `results/confirmatory.json` | code yes; Stage 1 adapters via the models collection | reward result, Stage 1 only; stress test reverses it (`results/a10/v2_env.md`, PCSTR-d 0.176 vs tool_trust 0.706) |
| RSC (recovery-start curriculum; earlier name "RS-GRPO") | `actionproof/stage3/{tasks,rollout}.py` | training-start distribution: half the pool starts from scripted fault states; pool adds fault classes | yes, 5 seeds (`runs/s3_rv_mix_s0..4`) | S3 H1/H2 (vs A12 R_V); S3-A1/A2 (vs class-matched control) | recipe: supported (+0.206 [0.168, 0.247], 5/5, `results/s3/s3_summary.json` `splits.s3_conf_rec.across.vsc`). Start-state part: H3′ supported (Holm 0.036); H4′p/H4′d not (Holm 0.111/0.337) (`results/s3/s3_addendum2.json` `part_a.tests`) | yes | adapters published | **the only training method with a registered confirmatory gain; it is a data distribution, not an optimizer** |
| RSC 25% mix | `stage3`, `--rec-frac 0.25` | recovery-start share 0.25 | dev only (`s3_rv_mix25_s0`, `_s1`; s1 incomplete per run manifest) | S3 development selection rule (`docs/stage3/RESEARCH_REPORT.md` §6) | lower on `s3_rec_dev` than 50% (0.553/0.667 vs 0.807/0.807, `results/s3/dev/*.summary.json` greedy vsc) | none | dev files only | not carried forward |
| Safety-first lexicographic reward (A) | `actionproof/stage3/rewards.py` `safe` | reward ordering: violation < false claim < honest failure < success | dev pilot, 1 seed, 100 steps (`s3_p_safe_b_s0`, `s3_p_safe_mix_s0`) | S3 pilot selection rule (§6) | `s3_rec_dev` VSC 0.553 (alone), 0.540 (with recovery starts) vs 0.633 for recovery starts with R_V | none | dev files | not carried forward (`docs/CONSISTENCY_AUDIT.md` #1) |
| Lagrangian reward (D) | `stage3/rewards.py` `LagrangianReward` | adaptive multiplier on violations | dev pilot, 1 seed (`s3_p_lag_mix_s0`) | same | `s3_rec_dev` 0.560 | none | dev files | not carried forward. Scalarises before normalising, a known weakness (RESEARCH_REPORT §7) |
| Potential-based shaping (C) | none | — | **never** | analytic | invariant shaping cancels in a group baseline shared by every rollout of a prompt (RESEARCH_REPORT §4C) | none | — | not trained (correct argument) |
| ACK starts (addendum 4) | `stage4/train_variants.py` | recovery starts end at the acknowledgement; scripted read removed | yes, 5 seeds (`s3_ackonly_s0..4`) | S3-A4 D1, D2 | neither supported (Holm 0.206, 0.262; `results/s3/s3_addendum4.json` `primary`); reading "inconclusive" | registered, negative | adapters on cluster only | negative |
| Rebalanced RSC (addendum 2B) | `stage3/tasks.py` rebalanced pool | receive-then-pay partials 6 → 21 of 400 | yes, 2 seeds + 2 control | S3-A2 Part B reading table | receive-then-pay repair 0.005 (W ≤ 0.15 → "not explained by pool frequency"; claim S3-A2B) | `results/s3/s3_addendum2.json` | adapters published | negative for frequency |
| Hidden-prefix "STATE" arm (addendum 3) | `stage4/train_variants.py --hide-prefix` | prefix replayed into the simulator but hidden from the conversation | **never** | — | withdrawn before training as degenerate (`docs/research/ADDENDUM3_WITHDRAWN.md`; red team §3.10) | none | — | withdrawn |
| SR1 tool-surface randomisation | `stage4/surface.py` | per-episode random tool and argument names | yes, 3 seeds (`s3_sr_mix_s0..2`) | SR-H1, SR-H2 | both not supported: +0.00 [−0.04, 0.04]; −0.248 on canonical tools (`results/s4/sr1_summary.json`) | registered, negative | cluster only | negative |
| FCA (forked counterfactual action credit) | `stage4/{advantage,forks,rollout_fca,train_fca}.py` | per-token advantages: forced canonical branches at decision states replace 4 of 8 rollouts | dev only, 2 seeds (`s4_fca_s0`, `_s1`) | kill criterion fixed before training (`docs/research/algorithm/FCA_GRPO.md` §7): ≥ +0.03 on `s3_rec_dev` | **failed**: −0.097 (0.640/0.780 vs 0.807; `results/s4/fca_dev_summary.json` `per_seed`) | none (never on test) | adapters published | **dropped** |
| C5 verified-progress turn credit | `stage4/{progress,rollout_c5g,train_c5g}.py` | per-turn advantage from a verifier-defined potential | dev only, 2 seeds | kill criteria (`docs/research/algorithm/C5G_DEV.md`) | failed: −0.32 `s3_rec_dev`, −0.37 `proc_dev` (`results/s5/s5_dev_summary.json`); read turns get negative credit by construction | none | cluster only | dropped |
| G sampler-only reconciliation hint | `stage4/rollout_c5g.py` mode guide | hint in ~5% of training rollouts | dev only, 2 seeds | same | failed: −0.11 / −0.033 | none | cluster only | dropped |
| AD-D1: GRPO / no-std / AWRB on the costed testbed | `ad/{train,rewards,rollout,worlds}.py` | AWRB: return replaced by its expectation over observation-equivalent worlds; no-std: `scale_rewards="none"` | dev 2 seeds, then 5 seeds for AD-C1 (`ad_{grpo,nostd,awrb,awrbn}_s0..4`) | AD-C1 (registered, frozen 99159bb + addendum 494ba06) | H_nostd +0.120 [0.088, 0.153] Holm 0.016, 5/5; H_awrb +0.192 Holm 0.016; H_awrbn +0.032 Holm 0.048; **H_AWRB (AWRB-centred vs no-std) −0.088, not supported** (`results/ad/c1_summary.json` `results`) | registered | cluster only | **no-std confirmed. AWRB adds nothing over plain centring** |
| AD-D2: EMA world-independent scale | `ad/{scale,train_ema}.py` | divides by one EMA scalar instead of the group std | dev 2 seeds (`ad_nostd_ema_s0,1`, `ad_awrbn_ema_s0,1`) | D2 gates G1–G4 | **failed G1** for both (`results/ad/d2_summary.json` `gates`) | none | cluster only | dropped |
| AD-D3: igc / IRC (uninformative-read local credit) on EMA scale | `ad/{readinfo,rollout_turns,train_irc}.py` | exact −c credit on reads that cannot change the posterior | dev 2 seeds (`ad_nostd_ema_irc_s0,1`) | D3 gates + mechanism prediction R1 | gates pass (+0.16 in both seeds), **mechanism prediction failed**: uninformative reads rose (drop share −1.79; `results/ad/d3_summary.json`) | none (H_D3 not run, AD-C1 note 3) | cluster only | not claimed |
| AWRB with read cost 0.30 | `ad/train.py --cost 0.3` | cost sensitivity | dev 1 seed (`ad_awrbc30_s0`, `ad_awrbnc30_s0`) | descriptive | in `results/ad/dev` | none | cluster only | descriptive |
| ad2 condition B (resampled hidden world per rollout) | `ad2/{rollout_rs,train}.py` | breaks the shared hidden state of a group | **never trained on the LLM**. Tasks 1301–1303 held for a `GROUPS` bug (`BRANCH_AUDIT.md` §3). Tabular only (`results/ad2/tabular_TB*.log`) | dev protocol hashes (`results/ad2/hashes_V2E1_CONDITION_B.json`) | tabular: GRPO c-insensitive in 97% of cells, wrong in 64% vs no-std 5.7% (FINAL_ALGORITHM_REPORT TB1) | none | tabular reproducible on CPU | tabular theory evidence only |
| LAB1 CDVO family (tabular) | `results/lab/`, `ad2/tabular` | critic-based decision-value objective | tabular 5 seeds | LAB1 H1/H2 (registered, 1bbbdea) | **not supported** (CDVO worse than the SFB baseline in 0/5 seeds; claim LAB-L1); normaliser intervention H4 supported (LAB-L2) | registered, negative | CPU | rejected |
| P3 Dn (no-std applied to P1's D) | `actionproof/p3/train_nostd.py` | `scale_rewards="none"` on D's recipe | **running** (tasks 1610–1612) | `PREREGISTRATION_P3.md` (7074e8d) | pending | pending | — | main session |
| "Recovery-State Adaptive GRPO" (difficulty-adaptive recovery-start sampling; the user's pasted first choice) | **none**. Every pool is fixed (`ad/pools.py`, `stage3/tasks.py`). No sampler adapts to reward variance or difficulty | — | **never** | — | — | none | — | not implemented. Nearest existing evidence: the fixed 25% vs 50% development pilot |

## 2. What the table establishes

1. **Only two training-side changes have registered confirmatory support**:
   - RSC, a start-state data distribution;
   - removing the group std (AD-C1 H_nostd), a known estimator (Dr. GRPO's advantage).

   Every new estimator built on top failed its own pre-set rule or its registered test: FCA, C5, G, D2, D3's mechanism, AWRB over no-std, and CDVO.
2. **RSC is a curriculum over start states**, closest to restart distributions and reverse curricula (Florensa et al. 2017, arXiv 1707.05300; Go-Explore, Ecoffet et al. 2019, arXiv 1901.10995). It is not a new optimizer. The write-up already claims no novelty for it (WRITEUP §7 "Training from failure states").
3. **The start-state part of RSC is only partly separated from fault coverage.** Against the class-matched control:
   - recovery-start VSC: supported (H3′);
   - full-episode partial and duplicate classes: not supported (H4′p, H4′d).
4. **ACK starts did not separate scripted evidence from start position** (S3-A4 inconclusive).

### The FCA red-team critique, checked against the code

`actionproof/stage4/rollout_fca.py` logs a forced branch "with sampling log-probability 0".

With `vllm_importance_sampling_mode = sequence_mask`, TRL then weights each forced sequence by exp(Σ log π_old(token) − 0). That is the probability of the exact forced text (`grpo_trainer.py` lines 2229–2241 in TRL 1.7.1).

The critique therefore holds as stated (`docs/RESEARCH_RED_TEAM_REVIEW.md` §3.7):
- the forced term reweights actions by their current probability, so it adds little signal for rare repairs;
- half of the on-policy samples were replaced.

FCA's negative result is about this implementation, not counterfactual credit in general.

The forced-branch weights are reported by the red team (sampler/trainer log-prob gap > 10 nats in 184 and 180 of 200 steps). I re-read them in `results/audit/redteam_recompute.json` `train_logs.s4_fca_s{0,1}`.

## 3. Missing comparisons, and whether they fit before Nov 28

| Missing | Why it matters | Cost | Fits with analysis time? |
|---|---|---|---|
| Adaptive vs fixed recovery-start share at equal budget | the user's candidate. The only evidence is one 25%-vs-50% dev pilot | ≥ 3 arms × 3 seeds × ~2 h + eval, plus a registration | **No**: lanes are taken by P3 and the judges until about Nov 26 evening, and there is no dev evidence to pick a schedule |
| DAPO dynamic sampling (drop zero-variance groups and refill) | 0.48–0.82 of groups carry no gradient (`OPTIMIZER_AUDIT.md` §3) | 3 seeds on D's recipe | No (same reason). Theory predicts it lands where GRPO does (TH1 descriptive, 720/721) |
| SELF history starts (red team §4.3) | the one design that separates start position from oracle history | 5 seeds + new rollout code | No |
