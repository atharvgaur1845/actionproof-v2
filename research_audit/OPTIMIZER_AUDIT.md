# Optimizer audit: what update is actually computed (addendum §4)

2026-11-25. Sources:
- the trainer code: `actionproof/stage3/train.py`, `actionproof/train/grpo_proc.py`; AgentTune `TrlAgenticGrpo` at d4db4c1;
- TRL 1.7.1 `trainer/grpo_trainer.py` and `grpo_config.py`, read on the cluster from `.venv`;
- each run's saved `runs/<run>/grpo_training_config.yaml`;
- every `runs/*/checkpoint-*/trainer_state.json`. A Slurm scan (`scripts/audit/redteam_recompute.py`) wrote `results/audit/redteam_recompute.json` key `train_logs`.

Versions: TRL 1.7.1, transformers 5.22.1, vLLM 0.23.0, peft 0.21.2, torch 2.11.0 (`requirements.lock`).

## 1. Configuration actually used

The values were read from the saved YAML of seven runs: Stage 1 PCSTR-d, A12 R_V, S3 RSC, P1 D, AD GRPO, AD no-std and P3 Dn. They are identical in every run, except `scale_rewards`.

| Setting | Value | Effect |
|---|---|---|
| `num_generations` | 8 | group size |
| `generation_batch_size` = `gradient_accumulation_steps` = `steps_per_generation` | 32 (4 prompts × 8) | one optimizer step per batch of rollouts |
| `num_iterations` | 1 | each batch is used once |
| `epsilon` (low = high) | 0.2, `delta` null | PPO clip range |
| `loss_type` | `dapo` | sum over all completion tokens of the batch, divided by the token count (`num_items_in_batch`) |
| `scale_rewards` | `group` (every run except the no-std arms, where it is `none`) | group std division |
| `beta` | 0.02 | k3 KL penalty to the reference (the warm start) |
| `importance_sampling_level` | `token` | — |
| `vllm_importance_sampling_correction` | true, mode `sequence_mask`, `clip_max` 3.0, `clip_min` null | vLLM-vs-trainer mismatch correction |
| `use_bias_correction_kl`, `off_policy_mask_threshold` | false, null | — |
| tool / environment tokens | `env_mask` from the rollout → TRL `tool_mask` (`grpo_trainer.py` l. 1914–1916) | tool-result tokens carry no loss |
| `temperature`, lr, LoRA | 1.0, 2e-5, r16 | — |

## 2. What the loss reduces to

With one optimizer step per rollout batch, `old_per_token_logps` comes from the same parameters as the update. TRL computes it explicitly because vLLM is used (l. 2694–2699).

So the PPO ratio ρ = exp(log π − log π_old) is 1 up to numerical noise, and the clip `min(ρA, clip(ρ, 0.8, 1.2)A)` is never active.

**The logs confirm it.** `clip_ratio/region_mean` is exactly 0.0 at every logged step of all 120 GRPO runs that log it:
- Stage 1;
- A12, K1, ID1;
- S3 and its addenda, SR1;
- FCA, C5, G;
- AD;
- P1 D, P2 E.

Source: `results/audit/redteam_recompute.json` `train_logs.*.clip_region_max`.

The per-step objective whose gradient is computed is therefore

  L(θ) = (1/N_tok) Σ_i Σ_t m_it · [ −w_i · Â_i · ρ_it(θ) + β · KL3_it(θ) ],   ∇ρ_it = ∇ log π_θ(y_it | ·) at ρ = 1,

where:
- Â_i = (R_i − mean_g R) / (std_g R + 10⁻⁴) with Bessel std. Under `scale_rewards="none"`, Â_i = R_i − mean_g R;
- m_it masks prompt and tool tokens;
- N_tok is the number of model tokens in the 32-rollout batch;
- w_i = exp(Σ_t m_it (log π_old − log π_vLLM)) is the sequence-level sampler-mismatch weight, set to 0 when w_i > 3;
- KL3 = exp(r − l) − (r − l) − 1, with r the reference log-prob and l the policy log-prob.

**In words: on-policy, group-mean-baselined REINFORCE with per-group std scaling, token-level (DAPO-style) aggregation, a k3 KL penalty and a masked sequence-level importance weight for the vLLM/trainer mismatch.** The clipped surrogate is present in the code but inert.

The mismatch weight is close to 1 in every main run: batch-mean weight 0.96–1.01 for A12, S3, P1, P2 and AD. Masked sequences occur in 0–42 of 200 steps. FCA is the exception: mean weight 0.78–0.79, a log-prob gap above 10 nats in 180–184 of 200 steps (§4).

## 3. Signal actually available

| Run family | Mean share of groups with zero reward variance (no gradient) | Mean token entropy |
|---|---|---|
| A12 R_A | 0.78–0.81 | 0.012 |
| A12 R_V | 0.58–0.62 | 0.019–0.021 |
| S3 RSC | 0.44–0.51 | 0.029–0.033 |
| class-matched control | 0.49–0.65 | 0.019–0.022 |
| ID1 | 0.98 | 0.010 |
| **P1 D** | **0.78–0.82** | 0.008–0.009 |
| P2 E | 0.80–0.82 | 0.009 |
| AD GRPO | 0.65–0.69 | 0.011–0.012 |
| AD no-std | 0.07–0.10 | 0.015–0.016 |

Source: `train_logs.*.frac_zero_std_mean`, `entropy_mean`. Ranges are over seeds.

Two consequences:
- **P1 D's RL stage receives gradient from fewer than a quarter of its groups.** Its warm start already solves most training prompts, which bounds what GRPO can add (P1_DECOMPOSITION §4).
- **ID1's null is mechanical.** At 0.98 zero-variance groups and no sampled read (`results/id1/id1_summary.json` `id1_rv_s0.train_read_rate`), the reward never sees a read.

## 4. Each method's objective, compared with the literature (each source verified at arXiv, 2026-11-25)

| Method here | Advantage | Aggregation | Ratio / clip | Sampling | Closest literature | Material difference |
|---|---|---|---|---|---|---|
| "GRPO" (every arm unless stated) | group-mean centred, group-std scaled | token-level over the batch | ratio ≡ 1, clip inert | 8 on-policy rollouts per prompt, all kept | GRPO, Shao et al. 2024 (arXiv 2402.03300). Its GRPO setup states "the policy model only has a single update following each exploration stage", so the ratio is also 1 at the first inner step | GRPO's original loss averages per sequence (1/\|o_i\|); TRL's `dapo` aggregation does not. Otherwise this is the original single-update GRPO. **The name is accurate if the single update and aggregation are stated** |
| no-std arm (AD-C1 "Dr. GRPO", P3 Dn) | centred only | token-level over the batch | ratio ≡ 1 | same | Dr. GRPO, Liu et al. 2025 (arXiv 2503.20783): removes the std **and** the 1/\|o_i\| length normalisation, using a constant normaliser | **Only the std removal is Dr. GRPO's.** The aggregation stays DAPO's token mean. The two differ by a per-step positive scalar, not per sequence (appendix_v3 §A already states this). Call it "GRPO without std scaling (Dr. GRPO's advantage)" |
| centred advantage, in general | R_i − mean_g R | — | — | — | RLOO, Ahmadian et al. 2024 (arXiv 2402.14740): leave-one-out baseline, and "clipping is rarely necessary in RLHF" (clipped < 5% of the time) | The group mean includes i itself; it equals the leave-one-out baseline up to the factor G/(G−1) (stated by Dr. GRPO) |
| DAPO (what is and is not used) | — | token-level ✓ | clip-higher ✗ (ε_high null) | dynamic sampling ✗ (zero-variance groups kept: §3) | DAPO, Yu et al. 2025 (arXiv 2503.14476): Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, Overlong Reward Shaping | Only one of DAPO's four techniques is in use. Do not call any arm "DAPO" |
| RSC | as GRPO | as GRPO | as GRPO | **start-state distribution**: half the prompts begin from scripted fault prefixes | reverse curriculum, Florensa et al. 2017 (arXiv 1707.05300); return-then-explore, Go-Explore, Ecoffet et al. 2019 (arXiv 1901.10995) | A data/curriculum change on top of unchanged GRPO. No adaptivity (fixed 50%) |
| AWRB | E[R \| trajectory] over observation-equivalent worlds | as GRPO | as GRPO | as GRPO | Rao-Blackwellised baselines and returns (variance reduction) | Registered test: no gain over plain centring (H_AWRB −0.088) |
| FCA | per-token, forced branches replace 4 of 8 rollouts | as GRPO | forced branches logged with sampling log-prob 0, so w_i = π(exact forced text), masked if > 3 | forced plus on-policy | COMA; expected policy gradients; VinePPO (red team §3.7) | The forced weight is the current probability of the exact text, so it reweights and does not add rare-action signal (confirmed in the logs: gap > 10 nats in about 90% of steps) |
| C5 / D3-igc | per-turn credit (verifier potential; exact −c on uninformative reads) | as GRPO | as GRPO | as GRPO | per-step credit (GiGPO and similar, cited in WRITEUP §7) | development only; failed (C5) or mechanism failed (D3) |

## 5. Wording rule applied to the submission documents

- "GRPO" is kept: the configuration is the original single-update GRPO.
- Every document that describes the method now states the reduction once:
  - one optimizer step per 32 rollouts;
  - the clip is inactive;
  - so the update is the group-baselined, std-scaled policy gradient, with DAPO token aggregation, a KL penalty and a vLLM mismatch weight.
- The no-std arm is called "no-std GRPO (Dr. GRPO's advantage, DAPO aggregation)" at its first use in each document.

Edits are listed in the hand-back report and the commit message. No number changed.

**Not claimed:** that clipping would matter with more inner epochs; that DAPO's dynamic sampling would help. Both are untested here. TH1's descriptive simulation found that zero-variance filtering lands where GRPO does in 720/721 cells (`paper/main_v3.tex`, TH1 descriptive list).
