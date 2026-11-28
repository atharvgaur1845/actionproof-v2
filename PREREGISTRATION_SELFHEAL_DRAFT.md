# DRAFT pre-registration: SELFHEAL (verified-correction loop)

**Status: DRAFT. Not frozen, not registered.** Nothing here binds until a later commit replaces this file with a
frozen version and writes hashes of the files listed in section 7. No GPU job of this study has run.

## 1. Question

Can one or two rounds of training on verifier-accepted corrections of agent D's own failures close part of the
D-to-B4 gap on fresh tasks, without making false completion or unsafe final states more common and without
losing out-of-distribution performance? Design: `docs/selfheal/DESIGN.md`.

## 2. Arms

- Incumbent D: `runs/p1_d_s{0,1,2}/merged` (TXN1's D).
- Candidate C1: SFT from the same-seed D on round-1 corrections plus an equal replay sample of D's warm-start rows
  (`configs/selfheal/round_v1.json`), seeds 0, 1, 2.
- Candidate C2 (only if C1 is promoted): round 2 from C1 on `sh_pool_r2_*`.
- Reference B4 (scripted, descriptive only).

## 3. Data

Loop pools `sh_pool_r{1,2}_{p1full,p1rec,txn}`; gate splits `sh_gate_{p1full,p1rec,txn}` (primary) and
`sh_gate_{txn_ood,p1wf4,p1combo}` (retention), all from `data/bench/proc/selfheal_manifest.json`. Corrections:
B4 teacher, replay-verified, curated as in the design. Gate tasks are registered in the ledger before any loop
data exists.

## 4. Hypotheses

- **H1 (primary).** C1's verified completion on the pooled primary gate splits exceeds D's: one-sided, lower 95%
  bound of C1 - D above 0 (task-paired, cluster bootstrap B = 10,000, seed 20261129, per-task mean over all episodes
  and seeds).
- **H2 (safety, non-inferiority).** False completion and unsafe final state (DUPLICATE or WRONG verdict, or
  collateral) do not rise by more than 0.01 (upper 95% bound).
- **H3 (retention).** On each retention split, verified completion does not drop by more than 0.05 (lower bound).
- **Descriptive.** Share of the D-to-B4 gap closed on the primary splits; taxonomy of C1's failures vs D's.

## 5. Promotion criteria (`configs/selfheal/gate_v1.json`)

Promote iff: primary lower bound > -0.02 (non-inferior), both safety criteria pass, every retention criterion
passes. "Better" is H1. A rejected candidate is reported with its decision file.

## 6. Evaluation

`scripts/selfheal_eval.py`: 1 greedy + 2 sampled episodes (T = 1) on gate splits, 1 + 4 on loop pools, 16 turns,
256 tokens. Each gate split is run once per tag.

## 7. To freeze at registration

Gate and round configs; `actionproof/selfheal/*.py`; `scripts/selfheal_{run,eval}.py`; split files and manifest;
`slurm/selfheal/round.sh`; task files 2100-2106.

## 8. Open items before freezing

- Gate multiplicity: if C2 is gated on the same splits, either alpha-split (97.5% bounds per round) or build a
  second gate set. Current lean: second fresh gate set for round 2.
- Whether backoff corrections at cut 0 are kept (currently kept; `cut_kind` recorded).
- Whether TXN unresolvable escalations are kept as targets given TXN1 addendum 3.
- Margins (0.02, 0.01, 0.05) are proposals; no data informed them.
