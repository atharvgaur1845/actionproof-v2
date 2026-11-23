# AD-D3: uninformative-read local credit (IRC), development

Registered 2026-11-23 16:35 IST.
- **What had been seen:** AD-D1 seed-0 outputs and awrb seed 1. No grpo, nostd or awrbn seed-1 output, and no D2
  output.
- **Hash file:** `results/ad_hashes_d3.json` (`scripts/ad/freeze_d3.py`).
- **Untouched:** no D1 or D2 hashed file is edited.

## 1. Motivation (descriptive, development)

FAILURE_ANALYSIS F3: no arm learned selective verification. On cw_dev, 42–49% of every reading model's reads are
provably uninformative under the cw1 prior (`results/ad/read_waste_cw_dev.json`; `actionproof/ad/readinfo.py`). That
includes the free-read RSC and A12 baselines. Most are journal reads, which cw1 never faults, and re-reads after the
world is already identified. They cost 0.02–0.07 J_c per episode.

The candidate cause is credit smear. The sequence-level advantage gives a valuable read and a worthless read in the
same trajectory the same credit.

## 2. Change (one, relative to D2's nostd_ema)

The same return R_c is decomposed as

  R_c = [R_V − c · n_informative_reads] + Σ_{uninformative reads} (−c).

- **The bracket** is the episode-level return. It is centred in the group and divided by D2's EMA divisor s_t.
- **Each uninformative read's −c is credited only to its own turn's tokens**, as an added −c/s_t.
- **Definition.** A read is uninformative iff no prior world first diverges from the recorded trajectory at that
  read. Its observation was then fully determined by the prior and the history, so it carries zero information
  under the prior.
- **Informative reads and all other turns are untouched.**
- **No turn is normalised against other turns.** This avoids the C5 failure (normalised per-turn credit makes reads
  negative by construction; docs/research/algorithm/C5_RESULT.md).

**Honest statement of bias.** Crediting a turn's immediate cost only to that turn drops the causal effect of earlier
actions on whether the uninformative read happens. With respect to E[R_c] the estimator is biased in that respect. The
optimum is unchanged only under the assumption that an uninformative read does not change the continuation's value.
This is a development heuristic, registered as such.

Code: `actionproof/ad/train_irc.py` (`--estimator nostd_ema_irc`), `actionproof/ad/rollout_turns.py`,
`actionproof/ad/readinfo.py`. Tests: `tests/ad/test_irc.py`, `tests/ad/test_readinfo.py`. CPU dry run:
`scripts/ad/cpu_dryrun_irc.py`.

## 3. Runs

`nostd_ema_irc`, seeds 0 and 1:
- task 1216 after 1214, in chain A;
- task 1217 after 1215, in chain B.

Everything else as D2: cw_train, c = 0.05, warm start, 200 steps, 4 × 8 rollouts, the same learning rate, β and LoRA.

## 4. Gates and registered predictions (cw_dev, greedy, mean of seeds 0–1)

**Gates.** D1's G1–G4 against grpo seeds 0–1 (`scripts/ad/d3_gates.py`).

**Predictions** (descriptive, against nostd_ema seeds 0–1):
- **R1.** Uninformative reads per episode fall by ≥ 50% relative to nostd_ema.
- **R2.** Verify-after-pay does not fall by more than 0.05.
- **R3.** J_c ≥ nostd_ema's.

R1 and R2 together are the selectivity claim. R1 without R2 means the method suppresses reading in general.
