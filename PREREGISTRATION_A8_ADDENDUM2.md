# Pre-registration, Amendment 8 addendum 2: the X1 state cell and a second seed of the A7 acknowledgement arm

Registered 2026-11-14 20:30 IST, before either run below had started. At registration no `runs/grpo_tiny_a8x1s_*` or `runs/grpo_tiny_a7_ack_outcome_s1` directory existed. The commit time of this file is the registration time. Both runs were proposed by a cold external review (Fable 5.2, 2026-11-14). Every earlier pre-registration and hash file is unchanged. The new files are hashed in `results/a8c_hashes.json` by `scripts/freeze_a8c.py`.

## 0. Why

X1 (Amendment 8) has only the acknowledgement column of the channel × pool design. The read-back interaction is therefore reported as a bound. The state cell turns it into an estimate. The A7 acknowledgement arm has one seed; the state arm has two.

## 1. Runs

| Task | Run tag | Change from its reference | Script |
|---|---|---|---|
| 417 | `a8x1s_pcstr_d_noreport_s0` | X1 (414p) with the reward `pcstr_d_noreport` (the A7 state arm, task 411) instead of `ack_outcome`. Same 340-task list, warm start `rft_tiny_a7`, GRPO config, seed 0, 18 turns, time slicing, rollout log | `slurm/a8_x1s.sh` |
| 418 | `a7_ack_outcome_s1` | Task 412 at seed 1. Rollout log on | `slurm/a7_ack_s1.sh` |

Evaluation: every A6 split, greedy, through `slurm/a6_eval.sh`, as for X1.

## 2. Predictions (greedy `test_iid`, refund read-back = PWV_refund)

| ID | Prediction | Reads as |
|---|---|---|
| H-A8f | 417 read-back ≥ 0.90 | Removing the two classes does not change the state arm. Expected, and therefore not informative alone |
| H-A8g | Interaction (414p − 412) − (417 − 411) > 0, 95% paired task bootstrap CI excluding 0 (`scripts/a8_interaction.py`) | The class removal matters for the acknowledgement channel only |
| H-A8h | 418 read-back ≤ 0.20 | The A7 acknowledgement arm's 0.00 replicates in sign at seed 1. If it is above 0.20, the article reports the seed-0 value as one seed and gives both |

No result here changes the confirmatory analysis. 417 and 418 are one seed each. No between-seed inference is made.

## 3. Order and freeze

Queue order on the H200 pool: 417, then 418. Results that arrive after the experiment freeze (2026-11-15 18:00 IST) are not used in the submission and are reported as "not finished".
