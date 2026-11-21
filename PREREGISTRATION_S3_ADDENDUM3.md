# S3 addendum 3: is the recovery-start effect the start state or the scripted history in context?

Registered 2026-11-22, before any model named here is trained and before the outputs of S3 addendum 2 Part A (control seeds 3–4, task 966) are read. The registration time is the commit time of this file and `results/s3a3_hashes.json`. Earlier registrations and results are unchanged.

## 1. Question

A recovery start changes two things at once:
1. **the environment's state:** the world already holds the faulted write;
2. **the policy's context:** the scripted calls and their replies are in the conversation.

The class-matched control (addendum 1) removes both. The registered start-state effect (H3, H4) cannot say which one matters. That effect is on partial payments only, with 3 seeds (`docs/stage3/ACADEMIC_REVIEW_POST.md` §2.4). This was raised by the red-team brief and the post-results review.

## 2. Arms (one factor changes between neighbours)

| Arm | Environment starts in the faulted state | Scripted history in context | Runs |
|---|---|---|---|
| FULL: class-matched control `s3_matched` | no (full episode carrying the same fault) | no | seeds 0–4 (0–2: addendum 1; 3–4: addendum 2) |
| STATE: new, `s3_stateonly` | yes | **no** | seeds 0–4 (new) |
| HIST: RSC `s3_rv_mix` | yes | yes | seeds 0–4 (S3) |

- **STATE.** It trains on the RSC pool (`s3_trainb_half` + `s3_rec_train_half`). Each recovery task's scripted prefix is replayed into the environment but not into the conversation (`actionproof/stage4/train_variants.py --hide-prefix`), so the policy sees only the system prompt and the user request, as in a full episode. It meets a world where some requested steps already happened, possibly faulted. This is a real deployment case (an agent restarted without its memory), and the conversation stays truthful: nothing in it claims anything about the past.
- **Everything else matches RSC:** trainer, reward (R_V), warm start, settings, steps, seeds, code path (`actionproof.stage3.train`) and hardware (Slurm H200).

## 3. Hypotheses

The metric is VSC on `s3_conf_full`, partial-payment class (64 tasks): full episodes, identical for every arm, and the class where the start-state effect was found.

| | Contrast | Supported if |
|---|---|---|
| C1 | STATE − FULL | Holm-adjusted one-sided unpaired seed permutation p < 0.05 (252 assignments; Holm over C1 and C2), **and** the seed-averaged 95% task-bootstrap interval excludes 0 |
| C2 | HIST − STATE | same |

| Result | Reading |
|---|---|
| C1 yes, C2 no | The faulted start state is the active ingredient; the history in context adds nothing measurable |
| C1 no, C2 yes | The scripted history in context is the active ingredient |
| both | Both contribute |
| neither | Not separable with five seeds; reported as inconclusive |

**Secondary (descriptive):**
- the same contrasts on the duplicate class and on all of `s3_conf_full`;
- `s3_conf_rec`;
- recovery starts with the history hidden at evaluation (`scripts/s3_eval_hidden.py`, every arm);
- violations, stress, and the number of runs that learn partial repair (partial-class VSC ≥ 0.40).

**Statistics:** `scripts/s3_stats_a3.py`, using the S4-EVAL addendum 1 helpers.

## 4. Compute

- Tasks 970–974: training, Slurm H200 lane.
- Task 975: each new model once on the S3 confirmatory splits.
- Task 976: the hidden-history evaluation for every arm.
- Missing runs are reported as missing.
