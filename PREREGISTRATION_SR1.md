# SR1: does training across tool surfaces make the recovery behaviour survive a renamed API?

Registered 2026-11-22, before any model named here is trained. The registration time is the commit time of this file and `results/sr1_hashes.json`. Earlier registrations and results are unchanged.

## 1. Motivation (from registered results)

S4-EVAL G1 was not supported: on renamed tools, arguments and acknowledgements (`s4_api`), RSC minus A12 R_V is −0.016 [−0.044, 0.011]. Every trained model falls from about 0.70–0.83 VSC (English, canonical names) to 0.20–0.29. Trained agents still call training-time tool names in 13–35% of renamed-API episodes (`results/s4/s4_diagnostics.json`). The behaviour looks bound to the tool surface it was trained on.

## 2. Method (a known augmentation, not a new algorithm)

**SR = RSC + tool-surface randomisation** (`actionproof/stage4/surface.py`, `actionproof/stage4/train_variants.py --surface-random`):
- Six alternative surfaces rename every tool, every argument, the acknowledgement key and its ok, error and timeout tokens.
- **None shares a name with the held-out `s4_api` surface** (asserted at import and tested).
- Half of the (task, step) choices use the canonical surface. All eight rollouts of a GRPO group share one surface.
- The semantics, faults, state, verifier and reward are unchanged. The trace stays canonical, and the same actions give the same score under every surface (tested on 120 tasks per surface).
- Pool, trainer, reward (R_V), warm start, settings and steps are RSC's.

## 3. Runs

- `s3_sr_mix` seeds 0–2 (tasks 977–979).
- They train on the Slurm A100 lane, with copies on the H200 lane as a fallback. RSC trained on H200. **The training hardware differs, and this is disclosed.**
- All SR evaluations run on the H200 lane, as RSC's did.

## 4. Hypotheses

The comparison is SR seed s minus RSC seed s, s = 0–2, paired tasks.

- **SR-H1 (primary):** VSC on `s4_api` is higher for SR.
  - **Supported if** positive in 3 of 3 seeds **and** the seed-averaged 95% task-bootstrap interval excludes 0.
  - The unpaired 3-vs-3 seed permutation p is reported (smallest possible 0.05).
- **SR-H2 (non-inferiority, canonical surface):** on `s3_conf_full`, the lower bound of the seed-averaged 95% interval of SR − RSC is above −0.05.

**Secondary (descriptive):**
- the share of `s4_api` episodes that call a training-time tool name;
- every other S4 split (language, workflow, two faults);
- `s3_conf_rec`;
- violations.

**Not expected and not claimed:** any change on the unseen four-step workflow. Surface randomisation does not vary the step order.

## 5. Scripts

- `scripts/sr1_stats.py`.
- Evaluation: task 990, using the frozen S3 and S4 evaluation loops (`slurm/s3_conf.sh`, `slurm/s4_evalset.sh`).
