# Stage 5 development: verified-progress turn credit (C5) and realizable hint-guided sampling (G)

Registered 2026-11-23 09:58 IST, before any training. Branch `research/e1e2`. Development data only. A confirmatory study is
registered separately, and only for a candidate that survives §4.

## 1. Why these two (evidence on file)

- **"Refund, then stop" is common.** After a correct refund, the episode often ends without the re-pay, so episode-level credit
  penalises the correct first step (`research_mandate/REPORT.md` D5). TACA (arXiv 2611.11330 §3.4) describes the same
  credit failure for recovery calls in general.
- **A one-line reconciliation of visible amounts raises correct refunds sharply.** In E2 H2 (registered), correct
  refunds after a goods receipt rose by +0.48 [0.43, 0.52].
- **Novelty is not claimed for either method.**
  - C5 is turn-level credit with an intermediate reward (the MT-GRPO template; cf. GiGPO and RewardFlow), here keyed
    on the verified hidden state.
  - G is sampler-only hint guidance (PSP arXiv 2609.29051 Eq. 8; Guide-GRPO), here with a hint computable from the
    visible history.
- The question is whether either helps in this silent-failure setting at matched rollouts and steps.

## 2. Methods (code: `actionproof/stage4/progress.py`, `rollout_c5g.py`, `train_c5g.py`; tests: `tests/stage4/test_c5g.py`)

**C5.**
- Progress potential: Φ(s) = 1 − cost(s)/(2n), with the per-step cost taken from the verifier's step verdict on the
  authoritative state:

  | Verdict | Cost |
  |---|---|
  | VERIFIED | 0 |
  | ABSENT | 1 |
  | PARTIAL | 2 for pay, else 1 |
  | DUPLICATE | 1 |
  | WRONG | 2 |

- Per turn k: u_k = Φ(after) − Φ(before), and the local credit is l_k = Σ_{j≥k} α^{j−k} u_j, with **α = 0.5**.
- Token advantage on turn k's policy tokens = TRL's episode advantage + **ω = 1** · (l_k − group mean) / (group std + 1e-4).
  The mean and std run over all turns of all 8 rollouts of the prompt.
- Φ is used in training only; the policy never sees it.

**G.**
- In each prompt group, rollouts 2, 4, 6 and 8 (odd index) generate from a context in which every `list_payments` result
  showing live payments above 0 and below the request's stated total carries one line:
  "purchase order total T; live payments L; outstanding T − L" (all visible values).
- The training sequence is the un-hinted conversation.
- The hinted episode's policy tokens are re-scored under the un-hinted context, and those log-probabilities replace
  the sampling log-probabilities, so the loss is plain GRPO on the un-hinted context. This is PSP's choice, and a deliberate bias.

**Everything else** is identical to RSC (`s3_rv_mix`): pools `s3_trainb_half` + `s3_rec_train_half`, R_V, warm start
`runs/rft_proc`, 200 steps, 4 × 8 rollouts, lr 2e-5, β 0.02, T 1, LoRA r 16, 16 turns.

**Compute.**
- Same rollouts and steps as RSC.
- C5 adds CPU-side verdicts only.
- G adds one prefill pass per hinted episode.
- Generated and gradient-bearing tokens will be reported from the logs.

## 3. Runs

Tasks `slurm/tasks/1010`–`1013`: c5 s0, guide s0, c5 s1, guide s1. They run after the 2-step smoke test `1009`, which is not analysed.

## 4. Kill criteria (`scripts/s5_dev.py`), per candidate, against RSC seeds 0–1 (already on file)

| | Criterion | Result if met |
|---|---|---|
| Kill 1 | `s3_rec_dev` VSC not at least **+0.03** above RSC, averaged over seeds 0–1 | Dropped |
| Kill 2 | `proc_dev` VSC falls by more than **0.03** | Dropped |
| Kill 3 | KL at step 100 more than **10×** RSC's | Dropped |

A candidate that survives all three goes to a separate confirmatory registration:
- fresh hashed splits;
- at least 5 seeds;
- RSC and the class-matched control as baselines;
- primary endpoint sampled partial-payment VSC, plus the E1 machinery.

**Descriptive (not tests).**
- Per-class `s3_rec_dev` VSC.
- Partial-payment refund-and-new-key-re-pay share over training.
- The hinted-episode share.

Two seeds on development data cannot show that a method is worse in general. They show only whether it clears the bar set here.
