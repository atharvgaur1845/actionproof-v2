# P4: does randomising the demonstrations fix procedure replay?

Registered 2026-11-26, about 00:40 IST, before any P4 model is trained or evaluated. The augmented data
(`data/rft/p4_aug_rows.jsonl`) is the intervention, not an outcome. It is built by `actionproof/p4/build.py`, whose
code is frozen here, and its hash is added to `results/p4_hashes.json` by `scripts/freeze_p4.py --add-data`
before training starts.

## Why

- **The weakness.** Trained agents fail on renamed tools (S4 `s4_api`) and on the unseen four-step workflow
  (`p1_test_wf4`: D 0.486).
- **What we already know.** Recovery capability comes mainly from demonstrations (exploratory 2×2: SFT +0.344, then
  GRPO +0.140; `results/audit/dwarm_2x2.json`). SR1 randomised the tool surface *during GRPO* and failed.
- **What is untried.** Randomising the **demonstrations**. That is what P4 tests.

## Arms (identical recipe: `actionproof.train.rft`, 2 epochs, the settings of `slurm/tasks/1153_p1_dwarm.sh`)

| Arm | Data | Seeds |
|---|---|---|
| Dw | `data/rft/p1_dwarm_rows.jsonl` (4,961 rows; P1 arm D's warm start) | 0 (existing `runs/p1_dwarm`), 1, 2 |
| Dw_aug | `data/rft/p4_aug_rows.jsonl`: all of Dw's rows (half of the recovery episodes re-rendered on a surface) plus about 1,050 composition rows (about +21%) | 0, 1, 2 |

**Dw_aug's data.** It holds the same P1 recovery teacher episodes, with half of the tasks (fixed hash
`surface.choose`) rendered on SR1's synonym surfaces. None of those surfaces shares a name with the held-out S4 API
surface; `actionproof/stage4/surface.py` asserts this.

It adds oracle demonstrations on five new step sequences: order_receive, order_pay, pay_only, receive_only and
order_only. None contains order → receive → pay, and none is the four-step workflow. They use fresh ids, PO ids
and prompt text.

Every original warm-start row is kept. **No model selection; no hyperparameter changes.**

**Design change before registration.** The first build matched Dw's row count by dropping whole warm-start
episodes. It dropped 217 of 240, which would have changed the arm in a second way, so it was discarded before
this file was committed and before any training. Dw_aug therefore has about 21% more rows than Dw. The added rows
are the composition intervention itself. H1's surface change adds no rows.

## Evaluation

`scripts/p4_eval.py`: P1's loop unchanged, 1 greedy + 4 sampled episodes at T = 1.

| Split | Tasks | What it tests |
|---|---|---|
| s4_api | 150 | renamed tools (held-out surface) |
| s4_wf4, p1_test_wf4 | 100 + 60 | unseen four-step workflow |
| P1 test, five splits | 450 | canonical performance |

Dw seed 0's P1-test episodes are reused from `results/p1/test/Dw__*` (evaluated once, before this registration).

## Hypotheses and rules (`scripts/p4_stats.py`, committed now)

- **H1 (renamed tools):** Dw_aug > Dw on `s4_api`.
- **H2 (new workflow):** Dw_aug > Dw on `s4_wf4` + `p1_test_wf4`.
  - H1 and H2 are each supported only if the Holm-adjusted one-sided paired task bootstrap p < 0.05 **and** all
    three same-index seed differences are positive.
- **H3 (no canonical cost):** the lower 95% bound of Dw_aug − Dw on the P1 test is above −0.03.
- **Unit and statistic:** a model's task score is its mean sampled VSC; an arm's score is the mean of its 3 seed
  models; paired task bootstrap with B = 10,000, seed 20261126.
- **Descriptive (no test):** per-split VSC; on `s4_api`, the share of calls using training-time (canonical) tool
  names; greedy VSC.

## What would make the result an artefact, and the control

| Risk | Control |
|---|---|
| Augmented names leak into the test surface | asserted in surface.py; checked again in the stats output |
| More data, not better data | stated: Dw_aug has about 21% more rows, all of them composition demonstrations. H1 (surfaces) re-renders existing episodes and adds no rows. H3 checks canonical cost |
| Prompt overlap with test tasks | prompt-text disjointness asserted in the builder |
| Seed noise | the all-seeds rule |

A null result is reported next to SR1's.
