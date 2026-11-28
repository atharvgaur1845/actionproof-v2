# The Hidden World in the Group: how GRPO mislearns costly checking in tool agents

**One-line claim.** When all the attempts in a GRPO training group share the same hidden situation, GRPO teaches an
agent to check its work by a vote instead of by what checking is worth; removing the group standard deviation brings
the value back.

**Links (everything in one place).**
[HF collection: every model, adapter, dataset and Space](https://huggingface.co/collections/atharvgaur18/actionproof-everything-lexsi-2026-track-3-6b0b3bda152727794d53e5a3)
· [live demo](https://huggingface.co/spaces/atharvgaur18/actionproof-live)
· [paper (PDF)](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/paper/main_v4.pdf)
· [code](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate)
· [the paper's adapters](https://huggingface.co/atharvgaur18/actionproof-hidden-world-adapters)
· [episodes, splits and results](https://huggingface.co/datasets/atharvgaur18/actionproof-traces/tree/main/hidden_world)
· [video](https://github.com/atharvgaur1845/actionproof-v2/releases/tag/video-v4)
· [slides](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/docs/submission/ActionProof_deck.pdf)

**Tracks and problem statements covered**
- Track 3 (agentic fine-tuning): problem statement 9, measure, understand, fix.

> **Content warning.** None. Every task is a synthetic purchase order in a simulated procurement system. No harmful
> text, no real people and no real payments appear anywhere in this work.

## Abstract

AI agents that act through tools often get a reply such as "ok" that does not prove what happened. A payment tool can
say "ok" when nothing was paid, or time out on a payment that went through. A careful agent can check, for example by
reading the payment record back, but checking costs time.

We study what a popular training method, group-relative policy optimisation (GRPO), teaches an agent about when to
check. GRPO tries the same task several times (a group of attempts) and scores each attempt relative to the others by
subtracting the group's average reward and dividing by the group's standard deviation. In a tool environment, all
attempts at one task face the same hidden situation: the same database and the same silent fault. We call this the
group's hidden world.

We prove that GRPO's scoring then cancels out how much each hidden world matters. What is left is a vote: the agent
learns to check if most hidden worlds favour checking, however rare and expensive the others are. We registered our
predictions before testing them. In an exact analytical test of 731 held-out problems, the theory predicts GRPO's
learned decision in all 721 problems that have a clear answer; one registered hypothesis failed and one could not be
tested, and we report both. We then trained a 3.35-billion-parameter Cohere tool agent (Tiny Aya Global) five times
with each method. With standard GRPO it never learned to check, in all five runs; with the standard deviation removed
it learned to check and its verified completion rose from 0.687 to 0.807. A third registered test changed the price
of a check: GRPO still read nothing at a lower price and read freely only when checking was free, while the other
method read more as checking got cheaper. All five predictions of that test held, with one run per arm.

We do not propose a new optimiser and we do not claim GRPO always fails. We identify when it fails, why, and what to
use instead.

## Headline before → after

| Same model (Tiny Aya Global 3.35B), 300 unseen tasks | Verified completion | Checks per episode |
|---|---|---|
| Before: supervised warm start | 0.687 | 0.00 |
| After: GRPO, 5 runs | 0.687 (all 5 runs) | 0.00 |
| After: GRPO without the group standard deviation, 5 runs | **0.807** (+0.120, 95% CI [0.088, 0.153]) | 1.62 |

**Base model and method in one line.** Tiny Aya Global (3.35B) with LoRA, a supervised warm start on recovery
examples, then GRPO through AgentTune, with and without dividing by the group standard deviation.

> **For judges: one minute.**
> - **What is new:** a precise account of when GRPO's group scoring hides the value of checking: when the attempts in
>   a group share one hidden world. Proven, then tested in three registered experiments.
> - **Evidence:** 721 of 721 analytical predictions; the agent result above; and a registered cost test (5 of 5
>   predictions held).
> - **What we say plainly:** all five GRPO runs ended exactly at the warm start, so that comparison is weaker than five
>   varied runs would be; the cost test has one run per arm; everything is in a simulator.
> - **Try it:** the live Space runs the trained agent on a real GPU and reveals the database after each run.

## Problem statement and approach

### Why this problem: a real use case

Agents are starting to pay suppliers, refund customers and deploy software. Their tools sometimes reply "ok" for a
write that never landed, or time out on one that did. An agent that trusts the reply reports payments that never
happened; an agent that retries blindly pays twice. The fix engineers use is to read the record back before saying
"done". Reads are cheap but not free, so the right policy depends on how likely each hidden situation is and how
costly it is to miss it. We want agents to learn that trade-off, and most agent training today uses GRPO. So the
question is whether GRPO can learn it.

### Our approach, as a story

1. **Derive** what GRPO's update does when every attempt in a group shares one hidden world.
2. **Test the theory** on hundreds of exactly solvable problems, with predictions frozen before the test.
3. **Test it on a real agent**: train a Cohere model with GRPO and with the alternative, and compare.
4. **Change the price of checking** and see whether each method responds, as the theory says it should.

### Research questions

1. What does GRPO learn when every attempt in a group shares one hidden world?
2. When does that disagree with what checking is actually worth?
3. Do the predictions hold when groups are formed in other ways?
4. Does the same signature appear in a trained language agent, and does it respond to the cost of checking?

### Pre-registered hypotheses

Every experiment was registered in a dated file before any result existed, and changes were made only through dated
addenda.

- **Test 1** ([`PREREGISTRATION_TH1.md`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/PREREGISTRATION_TH1.md)): seven hypotheses about which decision each estimator learns (listed with
  their results below).
- **Test 2** ([`PREREGISTRATION_AD_C1.md`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/PREREGISTRATION_AD_C1.md)): the method without the group standard deviation ("std-free") beats GRPO on
  verified completion; eight theory predictions frozen in [`docs/research/AD_C1_THEORY_PREDICTION.md`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/docs/research/AD_C1_THEORY_PREDICTION.md).
- **Test 3** ([`PREREGISTRATION_AD_C2.md`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/PREREGISTRATION_AD_C2.md)): at a lower read cost GRPO still does not read, std-free reads and wins, and
  GRPO reads when reads are free.

### Threat model [T3]

The threat is the environment, not an attacker. Tool replies can be wrong: a phantom write answers "ok", a timeout
hides a write that did happen, and a write can be partial or duplicated. The agent never sees the true state. A
deterministic verifier, which reads the final database, decides whether the task was really done.

### Related work

- **Dr. GRPO** (Liu et al., 2025) removes the standard deviation to fix a bias toward easy prompts. We use its
  estimator; we do not claim it.
- **Bay and Yearick (2026)** relate GRPO, Dr. GRPO and DAPO through the group standard deviation for right-or-wrong
  rewards. **Fontana et al. (ACL 2026)** show that the AdamW optimiser ignores a global rescaling of rewards. Our
  effect is different: it is per hidden world and comes from the group, not from the optimiser.
- **Sakib (2026)** pairs rollouts on shared tool-fault schedules, which creates exactly our group structure, but
  studies variance rather than what the objective learns.
- **Calibrate-Then-Act** (Ding et al., 2026) prices exploration and reports GRPO collapsing to one policy across costs,
  without stating how its groups were formed.
- **LIMBO** (Li, 2026), **Evidence-Guided Recovery** (Zhu et al., 2026) and **Kaynar et al. (2026)** study duplicates
  and silent tool failure at evaluation or inference time. We study the training estimator.

## Hackathon repos (kits) used

- **AgentTune.** Trains every GRPO run with its `TrlAgenticGrpo` trainer. Its tool-call parser and prompt format are
  used in training and evaluation.
- **CuratorKIT.** Exact and MinHash deduplication (`ExactDeduplicator`, `MinHashDeduplicator`) of the 554 recovery
  demonstrations before the warm start. None was removed.
- **AuditKIT.** LLM-judge scoring (`TaskCompletion`, `AgentEvalRunner`) as a secondary check of episodes. It is never
  an outcome; the database verifier is.
- **CircuitKIT.** Tried for an interpretability probe; it did not run in our job environment, so it gives no result.
- **AlignTune, SafeTune.** Not used.

### How the repos connect

CuratorKIT cleans the demonstrations → the supervised warm start learns from them → AgentTune runs GRPO in the
simulator → the database verifier scores every episode → AuditKIT cross-checks a sample with LLM judges.

## Models and licenses

- **Base Aya model.** `CohereLabs/tiny-aya-global`, 3.35B parameters, bf16, fine-tuned with LoRA adapters (rank 16,
  alpha 32). License CC-BY-NC-4.0; every released adapter inherits it.
- **Other Cohere models.** Aya Expanse 8B and 32B (CC-BY-NC-4.0) appear only as untrained comparison agents and as a
  judge.
- **Closed-model baseline.** None in this work's results.

## Datasets, benchmarks and licenses

- **Source of data.** Synthetic, generated by ProcureSim, our in-memory procurement simulator: supplier orders, goods
  receipts, payments and journal entries. Each task carries one hidden fault plan; 60% of tasks are clean.
- **Languages covered.** English.
- **Train / test splits, leak check.**

  | Split | Size | Use |
  |---|---|---|
  | `cw_train` | training pool | GRPO training |
  | `cw_conf` | 300 tasks (121 faulted) | Test 2 evaluation |
  | `cw_c2` (sealed) | 300 tasks (120 faulted) | Test 3 evaluation |
  | Test 1 held-out suite | 731 analytical problems | Test 1, frozen by hash |

  [`scripts/ad/c2_split.py`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/scripts/ad/c2_split.py) asserts that no task identifier appears in two splits (a direct-match leak check).
- **General capability benchmark [T3] (retention check).** For the earlier GRPO agents of this project we measured
  IFEval and BFCL (English): no measurable loss against the warm start, for example IFEval +0.000 [−0.016, +0.017]
  ([`results/r1/r1_summary.json`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/results/r1/r1_summary.json)). The agents of this article were not re-tested on them.
- **CuratorKIT filtering.** 554 demonstrations before, 554 after (no duplicates found).
- **Tool / task environment, trajectories [T3].** ProcureSim exposes read tools (with an optional consistent read),
  write tools with idempotency keys, a hand-off to a human and a final report. Every training and evaluation episode
  is released as a trace (see Artifacts).

## Metrics, and why these

| Metric | Better | Why |
|---|---|---|
| Verified safe completion (VSC) | higher | The task really happened, with no duplicate, and the agent's report is true. Read from the database, not from the agent. |
| False success | lower | The agent said "done" when it was not. This is the failure that hurts businesses. |
| Checks (reads) per episode | matches the value | Checking should track what it is worth: not zero, not everything. |
| Tool-call validity | higher | Every call parses and is a real tool. |
| Wall time and latency | lower | Cost of running the agent. |

- **What should not get worse.** VSC on clean tasks: checking must not break tasks that had no fault. It stays at
  1.000 for every method.
- **Checking the automatic judge.** The database verifier is the truth. LLM judges were compared with it, not with
  humans: the best judge given a database snapshot (Gemma 4 31B) agreed 0.989 of the time. No human labels exist.

### Agentic trajectory metrics [T3]

Test 2, 300 unseen tasks, greedy decoding, mean over runs:

| Method (runs) | Task success: all | faulted | clean | False success | Tool-call validity | Calls / ep | Reads / ep | Eval seconds / 300 tasks |
|---|---|---|---|---|---|---|---|---|
| Warm start (1) | 0.687 | 0.223 | 1.000 | 0.313 | 1.000 | 3.73 | 0.00 | 14 |
| GRPO (5) | 0.687 | 0.223 | 1.000 | 0.313 | 1.000 | 3.73 | 0.00 | 14 |
| Std-free (5) | 0.807 | 0.521 | 1.000 | 0.191 | 1.000 | 5.46 | 1.62 | 22 |
| AWRB (5) | 0.879 | 0.699 | 1.000 | 0.120 | 1.000 | 6.61 | 2.69 | 24 |
| AWRB, mean-only (5) | 0.719 | 0.302 | 1.000 | 0.279 | 1.000 | 4.31 | 0.56 | 16 |

Eval seconds are wall time for the whole batched split on one GPU. Batch-1 latency on the same base model was 2.44 s
per episode for the warm start (p95 3.05 s) and 3.82 s for an agent that checks (p95 4.68 s), on 40 simulator episodes
([`results/s2/latency_*.json`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/results/s2)). AWRB is an alternative return that averages over worlds the agent cannot tell apart.

## Method / experiments

### Training setup and hyperparameters

- **AgentTune / TRL GRPO.** 4 tasks × 8 attempts per step, 200 steps, learning rate 2e-5, KL weight 0.02, temperature
  1.0, up to 16 turns of 256 tokens each, gradient clipping 0.5, bf16. The only difference between methods is TRL's
  `scale_rewards` setting: divide by the group standard deviation (GRPO) or not (std-free). Read cost 0.05 per read.
- **Supervised warm start.** LoRA fine-tuning on recovery demonstrations deduplicated by CuratorKIT.
- **CuratorKIT.** Exact and MinHash deduplication with the toolkit defaults.
- **AuditKIT.** `TaskCompletion` judge, default prompt, run on a sample.

### Seeds

Five training runs (seeds 0 to 4) per method in Test 2; one run per arm in Test 3; ten seeds per problem in Test 1.

### Ablations

The estimator is the ablation: GRPO against std-free, plus AWRB and its mean-only version in Test 2, and ten estimators
in Test 1.

## Results

### Same model, before vs after

See the headline table above. The whole gain is on faulted tasks; on clean tasks every method scores 1.000.

### Finding 1: the theory predicts what GRPO learns (Test 1)

| Registered prediction | Result | Outcome |
|---|---|---|
| Theory predicts each estimator's final policy | median error 0.0003 | supported |
| Theory predicts GRPO's learned decision | 721 of 721 | supported |
| With shared worlds, GRPO follows the vote | 152 of 152 (77 of 77 where vote and value disagree) | supported |
| With independent worlds, always-check is still unstable | 67 of 69 | supported |
| Std-free follows the value | 661 of 661 | supported |
| GRPO loses no more than 0.01 against std-free | 53 of 75 (needed 90%) | **not supported** |
| Two kinds of correlated groups differ as predicted | 3 eligible pairs (needed 10) | **untestable** |

### Finding 2: in a real agent, GRPO never learned to check (Test 2)

All five GRPO runs read 0 times per episode and ended exactly at the warm start (0.687). The std-free runs read 0.93
to 2.02 times per episode and reached 0.807 on average: +0.120 over GRPO, 95% CI [0.088, 0.153] as registered, or
[0.075, 0.167] when runs are also resampled, Holm-corrected p = 0.016, better in all five pairs.

### Finding 3: GRPO's checking is all-or-nothing in the price; std-free's is graded (Test 3)

| Method | Read cost | Reads / ep | VSC | Faulted VSC | False success |
|---|---|---|---|---|---|
| GRPO | 0.05 | 0.000 | 0.707 | 0.267 | 0.293 |
| GRPO | 0.01 | 0.000 | 0.707 | 0.267 | 0.293 |
| GRPO | 0 | 2.723 | 0.883 | 0.708 | 0.117 |
| Std-free | 0.05 | 1.810 | 0.803 | 0.508 | 0.197 |
| Std-free | 0.01 | 2.630 | 0.883 | 0.708 | 0.117 |

Sealed split of 300 tasks, one training run per arm. Std-free beats GRPO at cost 0.01 by +0.113 [+0.077, +0.152], and
it reads 1.05 [0.95, 1.15] more times per episode at 0.01 than at 0.05. All five registered predictions held. The
theory explains the pattern: in a clean world a read only costs, so GRPO's vote against reading flips only when the
cost reaches zero.

### Confidence intervals and significance tests

Task-level bootstrap intervals (B = 10,000) and exact permutation tests with Holm correction, as registered. Test 2
also reports a two-level interval that resamples runs, then tasks.

### What didn't work (negative results)

- Test 1: GRPO's loss against std-free was not always small (53 of 75); one hypothesis could not be tested.
- Test 2: the GRPO arm did not move at all, so p = 0.016 is the smallest value the test can give, not five independent
  outcomes. A check made after the registered analysis shows GRPO did push checking down: under sampling the warm
  start reads 0.189 times per episode and every GRPO run 0.000 to 0.001.
- AWRB's mean-only version did not beat std-free (−0.088).
- Where reads are free, removing the standard deviation did not help (0.813 against 0.819).

### Sample input and output

A live run on the public Space, Task 0004, hidden world "phantom payment" (the payment tool says ok, but nothing is
stored):

| Turn | Script that trusts the reply | Trained agent (live, L4 GPU) |
|---|---|---|
| 1 | place order → ok | place order → ok |
| 2 | pay → ok | read the order back → placed |
| 3 | post journal → ok | pay → ok |
| 4 | report "completed" | read payments → **[ ] nothing was paid** |
| 5 | | pay again, same idempotency key → ok |
| 6 | | read payments → exactly one payment |
| 7–8 | | post journal, read it back |
| 9 | | report "completed", each step confirmed |
| **Verdict** | false success, no payment | goal met, every step verified |

### Cost / ROI

- **Compute used.** Training and evaluation ran on the BITS Pilani sharanga GPU cluster (one H100 or H200 per run;
  each training run up to about 2.5 to 5 hours), at no cost to us. Test 1 ran on CPU. Earlier project stages used HF
  Jobs credits, logged in [`docs/stage2/HF_CREDITS.md`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/docs/stage2/HF_CREDITS.md); LLM-judge calls cost $3.20 and $2.82.
- **Cost per 1k replies.** Serving the trained agent costs under $0.50 per 1,000 workflows on our own GPU
  ([`docs/upgrade/ROI.md`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/docs/upgrade/ROI.md)). No closed-model API was priced for this article.

## Limitations

- The exact theory covers one check decision; for multi-step workflows the main theorem still holds but gives no
  numbers.
- One model family, one simulator, English only.
- The GRPO arm of Test 2 did not move; Test 3 has one run per arm and one way of forming groups.
- Reads are counted whether or not they were informative.
- No human-labelled evaluation exists.
- The novelty builds on known facts about z-score scaling: what is new is treating the hidden world as part of the
  group and working out its exact consequences for a costly decision.

## With more credits

- **Planned approach.** The full design: 2 methods × 3 read costs × 2 ways of forming groups (shared and independent)
  × 5 runs, then a second Cohere model and a second simulator.
- **Expected before / after.** With shared groups, GRPO's read rate stays flat across positive costs (as in Test 3);
  with independent groups it starts to respond; std-free tracks the cost under both. These are predictions, not
  results.

## Artifacts

- **HF dataset.** [`actionproof-traces`](https://huggingface.co/datasets/atharvgaur18/actionproof-traces): folder
  [`hidden_world/`](https://huggingface.co/datasets/atharvgaur18/actionproof-traces/tree/main/hidden_world) holds this
  paper's evaluation episodes for Tests 2 and 3, training rollouts and logs, Test 1 results, the four splits, both
  papers and the registrations; the rest of the dataset holds the earlier studies' traces.
- **One collection with everything.** https://huggingface.co/collections/atharvgaur18/actionproof-everything-lexsi-2026-track-3-6b0b3bda152727794d53e5a3
  (73 items: every model, adapter, dataset and Space of the project).
- **HF models.** [`actionproof-hidden-world-adapters`](https://huggingface.co/atharvgaur18/actionproof-hidden-world-adapters):
  the warm start and all 23 adapters of Tests 2 and 3, one folder per run, with training logs and a model card. The
  trained agent shown in the demo: [`actionproof-agent-d`](https://huggingface.co/atharvgaur18/actionproof-agent-d).
  No GGUF or quantised builds were made.
- **GitHub repo.** https://github.com/atharvgaur1845/actionproof-v2 (branch `submission-candidate`).
- **Demo / Space.** [live demo](https://huggingface.co/spaces/atharvgaur18/actionproof-live) (real model on an L4 GPU) and the replay
  site https://huggingface.co/spaces/atharvgaur18/actionproof.
- **Video / slides.** https://github.com/atharvgaur1845/actionproof-v2/releases/tag/video-v4 and
  [`docs/submission/ActionProof_deck.pdf`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/docs/submission/ActionProof_deck.pdf).
- **Paper.** [`paper/main_v4.pdf`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/paper/main_v4.pdf) (6 + 6 format).

## Documentation and reproduction

- **Environment requirements.** Python with the repository's `requirements`; the checks below run on a CPU. Training
  needs one 80 GB GPU per run.
- **Repo layout.** [`actionproof/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/actionproof) (simulator, verifier, training code), [`scripts/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/scripts) (statistics, figures, checks),
  [`slurm/tasks/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/slurm/tasks) (one file per registered run), [`results/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/results) (every result file), [`paper/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/paper) (the paper), [`space_live/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/space_live)
  (the live demo), [`PREREGISTRATION_*.md`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/.) (registrations and addenda).
- **Run instructions.**
  - [`pytest tests/theory`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/tests/theory): the theory's tests.
  - [`python scripts/s2_numcheck.py`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/scripts/s2_numcheck.py): checks every reported number against its result file.
  - [`python scripts/ad/c1_stats.py`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/scripts/ad/c1_stats.py) and [`python scripts/ad/c2_stats.py`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/scripts/ad/c2_stats.py): the registered analyses.
  - [`python scripts/paper/v4_figures.py`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/scripts/paper/v4_figures.py): regenerates the figures.

## Declaration of work done outside the hackathon

None of this work predates 1 November 2026. Stage 1 (1 to 15 November) built an earlier simulator and reward studies.
Stage 2 (20 to 28 November) built ProcureSim, the benchmark, the trained agents, the theory and all three tests (Test 2
registered 23 November, Test 1 on 24 November, Test 3 on 28 November).

## Discussion: what we learned

In agent reinforcement learning, a training group is not just a prompt; it can be a prompt plus a hidden world.
Scaling rewards by that group's spread throws away exactly the information an agent needs to decide whether checking
is worth it. The fix is simple: subtract the group mean but do not divide by the group's spread (Dr. GRPO, RLOO), or
use a scale that does not depend on the group's hidden world.

**Implications.** Anyone training tool agents with GRPO on stateful environments should record which hidden world
each group shares, and should not rely on group standard-deviation scaling for decisions that involve paying for
information.

**Future work.** The full cost × group-law design, a second model family and simulator, and a human-reviewed
evaluation set.

## Teammates

Atharv ([@atharvgaur18](https://huggingface.co/atharvgaur18)), working solo.

## Author contributions

Atharv: research direction, hypotheses, approval of every registration before its run, and review of every claim.

## Acknowledgements

Lexsi Labs for the hackathon, its toolkits and HF credits; BITS Pilani HPC (sharanga) for GPU time; Cohere Labs for
the Aya models.

## References

- AgentTune: arXiv 2610.13335. AuditKIT: arXiv 2610.22511. CuratorKIT: arXiv 2606.21631.
- Tiny Aya: arXiv 2603.11510. Aya Expanse: arXiv 2412.04261.
- GRPO (DeepSeekMath): arXiv 2402.03300. Dr. GRPO: arXiv 2503.20783. RLOO: Ahmadian et al., 2024.
- The full list is in [`paper/refs_v4.bib`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/paper/refs_v4.bib).

## LLM usage statement

Claude (Anthropic) was used to help write code and to help with writing.

## Changelog

- **29 November 2026.** Rewritten around the main paper; adds Test 3 (cost test) results. Replaces the Stage 1
  article, kept as [`docs/HF_ARTICLE_stage1.md`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/docs/HF_ARTICLE_stage1.md).

## Appendix

- **Prompts.** The agent's system prompt is [`data/bench/proc/system_prompt.txt`](https://github.com/atharvgaur1845/actionproof-v2/blob/submission-candidate/data/bench/proc/system_prompt.txt); the judge prompts are under
  [`evaluation/llm_judge_committee/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/evaluation/llm_judge_committee).
- **Extra tables.** Per-run results for Test 2 are in [`results/ad/conf_summaries/`](https://github.com/atharvgaur1845/actionproof-v2/tree/submission-candidate/results/ad/conf_summaries); Test 1's ten estimators are in the
  paper's appendix.
