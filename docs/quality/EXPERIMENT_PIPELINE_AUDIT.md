# Experiment pipeline audit

2026-11-24, branch `quality-upgrade`. Every statement below comes from a command run today; the commands are listed per section.

## A. Registration integrity

**Command.** Every `scripts/freeze_*.py --check` was run on the cluster copy, where all frozen data files exist (44 registrations).

| Outcome | Registrations |
|---|---|
| Pass | A4, A6, A7, A8 (+ addenda b, c), A10b, A11, A12 (+ b, c), A14 (+ b), A15, A17, bench, C5G, E1E2, ID1 (+ D1), J1 addendum 1, K1 (+ models), models, P1, P2, R1-D1, R1-T-D2, S3 (+ addenda 1–4), S4-EVAL (+ A1), SR1 |
| Mismatch, superseded by a documented deviation freeze that passes | R1 → D-R1-1; R1-T and R1-T-D1 → D-R1T-2; J1 → J1 addendum 1 (`committee.py` gained the v0 variant, commit bd60beb, registered before any v0 output) |
| Mismatch, documented deviation **without** its own freeze | A5: `verifier_a5.py` and `a5_rewards.py`, D-A5-3 (e499655, crash fix for malformed certificate fields). A9: `judge_audit.py`, D-A9-1 (23303d4, FP8-on-A100 output quarantined). Both predate the separate-freeze practice; the deviation is in `docs/stage2/DECISIONS.md` and the git history |
| Registration file absent on the cluster | A10, A13: present in git and locally, never synced. A sync gap, not an integrity failure |

**Result.** No frozen file changed without a logged deviation.

**Per experiment** (seeds, splits, primary outcome, decision rule, status, deviations): `docs/upgrade/INVENTORY.md` (`scripts/build_inventory.py`, 33 registrations, with their claims and result files), plus `PREREGISTRATION_P1.md` and `PREREGISTRATION_P2.md`.

## B. Run provenance (`scripts/run_manifest.py` → `results/run_manifest.json`)

**Defects found in the old manifest and fixed:**
1. It listed only runs with `final/adapter_model.safetensors`. That silently dropped:
   - **every C5 and G run**: `s5_*` save their adapter at the run root;
   - every crashed or incomplete run.
2. It had no family for C5, G, AD, P1, P2, ID1 or SR1.
3. It had no notion of completion: an adapter was taken to mean a finished run.

**New manifest.** One entry per run directory (124).

**Status needs evidence.**
- `complete` requires the trainer's own final log to reach the configured steps or epochs.
- A checkpoint alone never counts.
- `complete_unlogged` means a final adapter exists and its task did not fail, but no final log proves the count.

| Status | n | Runs |
|---|---|---|
| complete | 112 | |
| complete_unlogged | 4 | `rft_proc` (no run_config: trained on an HF job; the A12 registration fixes its settings), `rft_tiny_v2` (only the epoch-1 checkpoint state was kept; its `max_steps` 284 equals 2 epochs, and the final adapter differs from that checkpoint), `s3_p_lag_mix_s0`, `s3_p_safe_b_s0` (HF-job pilots) |
| incomplete | 1 | `s3_rv_mix25_s1`: checkpoints 25 and 50 of 200, no final adapter. **No result file or document uses it** |
| crashed | 2 | `grpo_tiny_a5_state_s0.crashed_D-A5-3` (D-A5-3); `rft_proc_v0` (its task failed; superseded by `rft_proc`) |
| running | 2 | `p1_d_s0`, `p1_d_s1` at build time |
| merged_copy / unknown | 1 / 1 | `a12_proc_rv_s0_merged`; `g0_grpo_smoke` (no config) |

**Fields recorded per run:**
- the producing task and its queue outcome;
- base model, initial adapter and its hash, final adapter and its hash, data hashes;
- seed, reward, lr, β, group size, prompts per step and rollouts per step;
- turns, completion length, and LoRA r / alpha / dropout / targets;
- configured vs logged steps or epochs, the log source, train runtime, checkpoints, and evaluation files.

**Not recorded by any run before P2, and listed as such rather than guessed:** GPU type, peak memory, tokens generated and trained, and the git commit at training time. The cluster copy is an rsync, not a git checkout, so code identity there is pinned by file sha256: registration freezes, and P2 provenance files.

## C. Evaluation provenance

**Before P2.** Episode records carry task id, sample id, decoding (implicitly, by script), the verifier's verdict and the full trace. They do **not** carry the split hash, verifier hash, adapter hash or expected counts. For registered studies those are pinned by the freeze files; for development runs they are not pinned.

**P2 onward.** `scripts/p2_eval.py` writes `<output>.provenance.json` with:
- the code sha256 of the evaluator, P1's evaluator, ledger, guard, verifier (`truth.py`) and environment;
- the split sha256 and task ids, and the adapter sha256;
- decoding (greedy T = 0; sampled T = 1.0, top-p 1.0, 256 tokens, 4 samples, seed `E1.seed_of(task, sample, turn)`);
- expected and actual episode counts.

**Coverage validator** (`actionproof/reporting/coverage.py`, tested in `tests/p2/test_p2.py`).
- It detects missing files, missing tasks, unexpected tasks, missing samples, duplicate episodes, missing verdicts and bad JSON. Unparsed turns and incomplete rollouts are counted as behaviour.
- P2's analysis refuses to run on an incomplete matrix.
- **Run today on all 70 P1 output files present: 0 defects.** Every file has 1 + 4 episodes per task. A few episodes per file end on an unparseable reply, counted as behaviour.

**Harness equivalence** (P2 smoke, task 1180). P2's evaluator, with ledger and guard off, reproduces P1's evaluator exactly on `p1_dev_rec`: 90 of 90 greedy and 360 of 360 sampled episodes have identical actions and verdicts (`logs/p2_equivalence.log`). So P2 and P1 numbers are directly comparable.

## D. Training-rollout instrumentation (C5, G)

**What exists.**
- C5 and G training logs (`results/s5/rollouts/*.jsonl.gz`, 6,400 episodes per run): step, task, the full trace (exact arguments and observations), the verifier's per-step verdicts, the start type, the per-turn progress u_k (C5), and the hinted flag (G).
- `runs/s5_c5_s*/c5_groups.jsonl`: per-group mean and std of local returns.

**What is missing:**
- token spans per turn, and the policy's raw text;
- sampling log-probabilities, the resulting token advantages, and the rollout index within a group (only implicit in log order);
- the policy-visible / evaluator-only split, which is implicit: the trace holds observations, `final_state` is evaluator-only.

These runs are dropped methods. Re-running them only to log more would spend GPU on unrelated algorithms, against the author's priority instruction. So the gaps are recorded here, and arm E's rollout log (`results/p2/rollouts/`) uses the same format.

**G is not an on-policy GRPO estimator.**
- `rollout_c5g.py` samples hinted rollouts from q = π(·|hinted context), then overwrites their sampling log-probabilities with p = π(·|un-hinted context).
- TRL's importance ratio is therefore 1 by construction, and the update estimates E_q[A ∇ log p] instead of E_p[A ∇ log p]. It is biased by the gap between q and p, and no correction is applied.
- The original q log-probabilities were not logged, so the gap is unmeasurable from the logs. `scripts/audit_g_mismatch.py` estimates it offline (KL(q‖p) per token and the importance-weight ESS on the 322 hinted episodes of G seed 0). It rebuilds assistant turns in canonical format, which is an approximation, because raw text was not logged.
- **Audit result** (task 1197, Slurm; `results/quality/g_mismatch_s0_ck100.json`; G seed 0 at checkpoint 100, 322 hinted episodes, 90,030 assistant tokens):
  - Per token the gap is small: the mean log q − log p is 0.0097.
  - Per episode it compounds: the mean summed log-ratio is 2.70 and the largest 21.8.
  - The implied importance weights p/q have an effective sample size of 1.01 out of 322 episodes (0.3%). A correctly reweighted estimate of the on-policy gradient from these rollouts would rest on about one episode. The update G applied therefore followed the hinted policy's tokens almost entirely, not the un-hinted policy it was meant to train.
  - Caveats: one seed and one checkpoint; assistant turns rebuilt in canonical format, because the raw text was not logged. This describes the estimator. It does not by itself prove that the mismatch caused G's development loss.
- G is dropped (−0.11, −0.033); **it must not be described as an unbiased on-policy method.**

**C5 credit assignment** (`actionproof/stage4/progress.py`).
- u_k = Φ(after turn k) − Φ(before turn k) is the state change caused by turn k's own tool call, so the credit goes to the responsible action.
- Normalisation, however, pools every turn of every rollout in the group. Read turns have u = 0 and get only α-discounted credit from later progress, so after centring they receive negative advantage whenever the group's mean local return is positive.
- `results/s5/` shows read turns at −0.5 to −0.7 in every block.
- Longer trajectories contribute more turns to the pooled mean and std, which is a length bias.
- These are why C5 penalises verification by construction (`docs/research/algorithm/C5_RESULT.md`).
