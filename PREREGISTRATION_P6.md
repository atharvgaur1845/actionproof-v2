# P6: does a robust demonstration warm start make surface-randomised RL work?

Registered: 2026-11-29, before any P6 training or evaluation.

## Question

P4 established that randomized recovery demonstrations improve the warm start (`Dw_aug`) on renamed tools and an
unseen four-step workflow. P4 did **not** test RL continuation. The earlier SR1 study randomized tool surfaces during
GRPO from the older `Dw` warm start and did not establish a gain on the held-out API. P6 tests whether surface-
randomised GRPO behaves differently when it starts from the stronger `Dw_aug` checkpoint.

This is a training-intervention study, not a new optimizer claim. It reuses the existing, tested surface randomizer in
`actionproof/stage4/surface.py` and keeps its contract: one surface per `(task, training step)`, with all eight rollouts
in each GRPO group sharing that surface. Half of the choices are canonical; the remaining choices use one of six
registered alternative surfaces. These training surfaces are disjoint from the held-out S4 API surface.

## Arms

Three matched seeds (0, 1, 2), paired by seed and by P4 warm-start checkpoint:

| Arm | Initial adapter | RL environment | Training entry point |
|---|---|---|---|
| `P6-CAN` | `runs/p4_dwaug_s{k}/final` | canonical ProcureSim API | `python -m actionproof.stage3.train` |
| `P6-SURF` | same `runs/p4_dwaug_s{k}/final` | `--surface-random` (50% canonical, 50% one of six alternate surfaces per task/step) | `python -m actionproof.stage4.train_variants --surface-random` |

All other settings are held fixed: Tiny Aya Global 3.35B; `R_V` state reward; pools
`data/bench/proc/s3_trainb_half.jsonl` + `data/bench/proc/s3_rec_train_half.jsonl`; 200 optimizer steps; 4 prompts × 8
rollouts; LoRA r16/alpha32; learning rate `2e-5`; KL coefficient `0.02`; temperature 1; max 16 turns; max 256 new
tokens; same P4 seed-specific parent adapter and matching training seed. No early stopping or checkpoint selection.

## Evaluation

Use the unchanged P1 evaluation loop via `scripts/p6_eval.py`: one greedy and four sampled episodes per task, T=1.

| Split family | Splits | What it measures |
|---|---|---|
| Held-out API | `s4_api` (150 tasks) | transfer to a tool/argument/acknowledgement surface not used by the P6 randomizer |
| Held-out workflow | `s4_wf4` + `p1_test_wf4` (160 tasks) | retention on the unseen four-step workflow |
| Canonical | five P1 test splits (450 tasks) | no regression on the original procurement distribution |

P4 `Dw_aug` outputs are reused as the SFT reference. These task splits were already evaluated for P4; therefore P6 is
a preregistered paired follow-up on existing locked tasks, **not** a fresh independent benchmark replication. No P6
test result may be used for model selection or another training run.

## Hypotheses and decision rules

A task score is the mean VSC over its four sampled episodes (the greedy episode is reported separately). Model-arm
scores average the three matched seeds. Paired task bootstrap: 10,000 resamples, seed `20261129`.

- **H1 — API transfer superiority:** `P6-SURF − P6-CAN` on `s4_api` is positive; supported only if the one-sided
  paired task-bootstrap `p < 0.05`, the 95% interval excludes zero, and all three matched seed differences are positive.
- **H2 — API retention:** `P6-SURF − Dw_aug` on `s4_api` is non-inferior with margin `−0.03`; supported iff the lower
  95% paired-bootstrap bound is greater than `−0.03` and every matched seed difference is greater than `−0.03`.
- **H3 — workflow retention:** same non-inferiority rule on `s4_wf4 + p1_test_wf4`.
- **H4 — canonical retention:** same non-inferiority rule on the 450-task P1 test union.
- **Safety guardrail (descriptive, not a separate superiority claim):** report false-completion, verifier violations,
  duplicates, tool calls, reads, and latency beside VSC for every arm/split. A higher VSC does not count as an
  unqualified improvement if false completion or invariant violations increase materially. "Materially" is fixed
  here as in `scripts/p6_stats.py`: on the 450-task canonical union, `P6-SURF − Dw_aug` false completion passes the
  guardrail only if its upper 95% bound and every matched seed difference are below `+0.02`.

No hypotheses are combined post hoc. Missing/corrupt task records fail the analysis; no silent dropping. Failed seeds
are reported, not replaced. Deviations require a dated addendum before looking at the affected output.

## Execution, compute and reporting cutoff

Each (arm, seed) runs as one HF Jobs H200 job (`scripts/p6_hf_launch.sh`, `slurm/p6_hfjob_run.sh`): the training task
`2300_p6_<arm>_s<k>`, then the evaluation task `2310_p6_eval_<arm>_s<k>` of the same arm and seed. Both tasks run
`scripts/freeze_p6.py --check` before doing work. All six jobs use one uploaded code bundle built from this registered
snapshot. Adapters and traces are uploaded to the private dataset `atharvgaur18/actionproof-jobs` and copied into
`results/p6/`; the statistics run only after all six jobs have finished. The evaluation task was split per arm (it was
one task per seed in the drafted package) before registration, so that no merged model has to move between jobs.

The hackathon submission closes on 2026-11-29 at 17:29 IST. If any of the six jobs has not finished, or the
statistics have not run, by then, P6 is not reported in the submitted papers; it stays in the "with more credits"
plan and is reported later in full, whatever the outcome. No partial or per-seed P6 result is quoted before the
registered analysis has run.

## Reproduction order

1. Install the files in this patch at the repository root.
2. Run `pytest tests/test_p6_stats.py -q` and the existing P4 freeze check.
3. After the P4 warm-start adapters and P4 evaluation outputs are present, run
   `python scripts/freeze_p6.py --register`; commit `results/p6_hashes.json` before training.
4. Run the six `slurm/tasks/2300_*.sh` training jobs.
5. Run the six `slurm/tasks/2310_*.sh` evaluation jobs (on HF Jobs, each follows its training task in one job).
6. Run `python scripts/p6_stats.py` only after every registered output exists.

A null or negative result is still a valid outcome. No new algorithm, general-purpose reliability guarantee, or
real-system claim is implied by P6.
