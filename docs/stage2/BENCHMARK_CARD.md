# ProcureSim benchmark card

ProcureSim is a separately implemented procurement task environment with five simulated services. It shares the ActionProof codebase, so it is not an independent external environment.

## Transaction semantics
See `docs/stage2/TRANSACTION_SEMANTICS.md`.

## Fault taxonomy
Multi-axis (state observability, contract coverage, recoverability, agent observability); see `PLAN.md` §G1.

## Data generation
`actionproof/bench/generator_proc.py` draws each task from a seed: a purchase order (supplier from 8, item from 10, quantity, unit price in integer cents), a workflow (`order_prepay`: order, pay, post; `receive_pay`: receive, pay, post; `pay_post`: pay, post), and a fault plan that places one fault on the first write of one requested step. Development, validation and confirmatory splits use the mix clean 20%, phantom 25%, partial 10%, lost acknowledgement 15%, timeout before commit 15%, never commits 15%; the GRPO training split `proc_train_b` uses clean 5%, phantom 35%, partial 15%, lost acknowledgement 25%, timeout before commit 5%, never commits 15% (Addendum 1). `proc_cf` (A14) has no generated fault: each task is run under five branch faults on the payment. Partial faults are placed only on `pay` and `receive`, the steps the oracle can repair. The prompt is written from the task alone and never from the fault plan.

## Split construction and leak checks

| Split | Tasks | Seed | Use |
|---|---|---|---|
| proc_demo | 240 | 101 | warm-start demonstrations only |
| proc_train | 400 | 102 | first gate (failed; kept for the record) |
| proc_train_b | 400 | 109 | GRPO training (Addendum 1: rebalanced towards classes where reading changes a reward) |
| proc_dev | 200 | 103 | development, checkpoint diagnostics |
| proc_val | 100 | 104 | development, checkpoint diagnostics |
| proc_conf | 300 | 105 | confirmatory, run once per frozen model |
| proc_combo | 100 | 106 | confirmatory: unseen two-fault combinations |
| proc_stress | 100 | 107 | confirmatory: duplicate delivery; lost acknowledgement with replica lag |
| proc_heldout_service | 100 | 108 | generated; exploratory only, not run |
| proc_cf | 250 | 110 | A14 forced-evidence forks (generated after its registration) |

Checks, each a test in `tests/integrity/test_procsim.py`: split files on disk equal the generator's output and task ids are disjoint across splits; phantom and partial acknowledgements are byte-identical to success; the prompt does not depend on the fault plan. Split hashes are frozen in `results/a12_hashes.json` and `results/a12b_hashes.json`. `scripts/eval_proc.py --confirmatory` appends every confirmatory run to an access log, and the gateway evaluation script refuses confirmatory splits.
## Verifier assumptions
See `docs/stage2/VERIFIER_SPEC.md`.
## Metric definitions
See `docs/stage2/METRICS.md` (definitions computed by code are in `VERIFIER_SPEC.md` §5–7).
## Known limitations
- One codebase: the simulator, the verifier and the gateway were written by the same team, so the simulator is "separately implemented", not independent. The reference verifier and the golden traces were written by separate agents from the specification only; agreement with them is agreement on independently specified cases, not ground truth.
- Faults are synthetic, placed one per task (two in `proc_combo`) on the first write of a step.
- Services are in-memory Python tables with monotonic versions; there is no network, no real latency and no concurrent writer.
- English prompts only; Aya models only.
## Intended uses
Studying how reward information shapes evidence acquisition; testing execution-verification layers against write-side faults.
## Prohibited interpretations
Results are not evidence about real payment systems, real suppliers, or models outside the Aya family. Agreement between our verifiers is not ground truth.
