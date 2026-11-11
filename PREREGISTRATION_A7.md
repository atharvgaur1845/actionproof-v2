# Pre-registration, Amendment 7: template inheritance and evidence response

Registered 2026-11-11, before any Amendment 7 GPU run. Design from the Fable 5.2 research review (SPEC §27, 2026-11-11). Deadline: 2026-11-15 midnight IST.

Amendments 0–6 and their hashes are unchanged:
- `results/a6_hashes.json` still covers `a6_rewards.py`, `a6_demos.py`, `generator_a6.py` and `a6_arm.sh`.
- New code lives in new modules, hashed in `results/a7_hashes.json` by `scripts/freeze_a7.py`.

## 0. Why

The A6 seed-0 results raised two anomalies.

| Anomaly | What we saw |
|---|---|
| Credit workflow | The state-reward arms stopped calling `apply_store_credit`. |
| Honest failure | It depended on the report terms in a way their incentives do not predict. |

Two environment and measurement facts were also disclosed before this registration:
- **D-ENV-1:** a cross-operation idempotency-key collision.
- **D-EVAL-1:** honest failure is measured only under a tool-schema shift.

**Thesis under test.** GRPO selects among whole trajectory templates installed by SFT. Behaviours the reward does not directly separate, here credit-workflow coverage and failure reporting, are inherited from the winning template.

## 1. Experiments

| ID | What | GPU | Status |
|---|---|---|---|
| E7-1 | Forced-prefix battery (§4) on existing A6 checkpoints | < 1 h | primary for the honesty question, exploratory |
| E7-2 | New warm start `rft_tiny_a7` = A6 rows + 20 credit demonstrations (§3), gate G-A7 | ~1.8 h | primary |
| E7-3 | GRPO arms on `rft_tiny_a7`: `pcstr_d_noreport` s0 and `ack_outcome` s0 | ~5.4 h | primary |
| E7-4 | GRPO arm M2c `pcstr_d_noFunder` s0 on `rft_tiny_a6` (§2) | ~2.7 h | secondary |
| E7-5 | `pcstr_d_noreport` s1 on `rft_tiny_a7` | ~2.7 h | conditional on H7-1 holding at seed 0; queued on hold |

Training configuration is identical to Amendment 6:
- 200 steps × 4 prompts × 8 rollouts, lr 2e-5, β 0.02, 18 turns
- pool `a4_train_B`, plain `EcomEnv`
- batched vLLM rollouts

Every arm is evaluated with `slurm/a6_eval.sh` (all A6 splits, greedy, neutral).

## 2. New reward arm (E7-4)

`actionproof/rewards/a7_rewards.py`:

    pcstr_d_noFunder = G + 0.2P + 0.1V - 0.5I - 0.3ND - 0.2D

It completes the 2 × 2 over PCSTR-d's report terms on the A6 warm start:

| Arm | ND | F_under |
|---|---|---|
| M1 `pcstr_d` | yes | yes |
| M2b `pcstr_d_noND` | no | yes |
| M2c `pcstr_d_noFunder` | yes | no |
| M2 `pcstr_d_noreport` | no | no |

## 3. New warm start (E7-2)

- **Pool:** `a7_credit_demo` (`actionproof/bench/generator_a7.py`). 20 fault-free credit-workflow tasks from the seed stream `actionproof-a7/credit-demo/41/{i}`, disjoint from every training and evaluation split. They run in `EcomEnvA4`, the only environment with the credit tool.
- **Demonstrator:** `actionproof/train/a7_demos.py`. It uses the A6 demonstrator's template, with refund key `refund-<order>` and up to 4 verified refund attempts, plus one `apply_store_credit` call with its **own** key `credit-<order>`, then audit, close and report. The credit is not re-read.
- **Warm start:** `rft_tiny_a7` = `data/rft/curated_a6.jsonl` (the exact A6 SFT rows) + the credit rows, trained with the same RFT command (2 epochs).
- **Schema note.** The 20 credit demonstrations carry the A4 system prompt, which lists the credit tool. The other rows carry the plain one.
- **Disclosed consequence.** `e_heldout` becomes "demonstrated in SFT, never rewarded in RL" for the A7 arms. That is the point of the test, and the result must be described this way.

**Gate G-A7** (`slurm/tasks/410_a7_init.sh`). Both conditions must hold, or no E7-3 arm runs:
1. The A6 gate (sampled HFR on refund-UR in [0.10, 0.70], 100 pool-B tasks, T = 1 × 4).
2. On `e_heldout`, greedy, the A7 init calls `apply_store_credit` with a key different from its refund key in **≥ 30%** of episodes. If the init does not do this, the inheritance test is uninformative.

## 4. Forced-prefix battery (E7-1)

`actionproof/eval/forced_prefix.py`.

**Prefixes.** Fixed conversation prefixes, identical for every model:
- 50 refund-UR episodes and 50 credit episodes, cut from `a6_pcstr_d_noreport_s0` evaluation traces.
- Types a–e as documented in the module.
- Types a–d are built in both schemas:
  - **a4:** as evaluated.
  - **plain:** the training system prompt, with no `credits` or `store_credits` fields.

**Readouts.** Exact renormalised probabilities of complete continuations:
- the report status, out of {failed, completed, escalated};
- the next tool name.

**Models.** The A6 init, every A6 arm's final adapter (both seeds where present), and checkpoints 50/100/150 of M1 s0, M2 s0 and M2b s0. After E7-3 and E7-4, the A7 arms are added.

**Committed prediction (Fable 5.2).** On type a in the plain schema, M1 and M2 both put P(failed) > 0.5.

## 5. Hypotheses and decision rules

**H7-1 (template inheritance; primary).**
- *Claim.* On `e_heldout`, greedy, the A7 state arm (`pcstr_d_noreport`, s0) calls `apply_store_credit` with a non-colliding key in ≥ 50% of episodes. The A6 state arms do so in 0%.
- *Kill.* Below 20%: template inheritance is not sufficient, and the paper keeps the workflow result as an observation only.
- *Secondary metrics:* credit committed (receipt), refunds re-issued after a read showing them, unsupported credit claims, close rate, and HFR.

**H7-2 (selection, not demonstration, removed the credit step).**
- *Claim.* The A7 ack arm's non-colliding credit-call rate is within ±10 pp of the A7 init's.
- *Kill.* If the ack arm's rate moves by more than 10 pp, a change under H7-1 cannot be attributed to the state reward's selection.

**H7-3 (template markers; CPU).** Per episode on `e_heldout`, the refund key format predicts credit omission with accuracy ≥ 0.9 across the A6 arms. Below 0.8, the two-template description is too coarse and is not used.

**H7-4 (report terms; secondary).**
- If M2c ≈ M2 (P(failed | stop-mode) ≥ 0.5) and M2b ≈ M1 ≈ 0, then "F_under, not ND, removes failure reports under the evaluation schema".
- If M2c ≈ 0, both report terms suppress the report.
- One seed each; reported per seed.

**Honesty decision rules.**
- **M2 seed 1** (A6, already queued). If its sampled P(failed | stop-mode) < 0.3, the M2 result is a single-run event and Finding 3 is dropped.
- **Battery.** If M1's plain-schema P(failed) on type a is > 0.5, "suppression" is replaced by "not robust to the schema shift".

**Statistics.**
- Task-level paired bootstrap (10k resamples) against the matched A6 arm, n = 200 for `e_heldout`.
- Wilson intervals per rate.
- No between-seed inference at k ≤ 2.
- Battery cells are means over 50 prefixes, with a bootstrap CI over prefixes.

## 6. What does not change

- The benchmark, the verifier and D-ENV-1: the environment is not fixed (disclosed in SPEC §27).
- The confirmatory test (task 400) runs as registered.
- M5 `tool_trust`, M3′ `outcome_G`, the a5lo arms and every A5 run stay on hold. None of them can change a conclusion.
