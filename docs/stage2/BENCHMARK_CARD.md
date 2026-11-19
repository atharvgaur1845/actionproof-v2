# ProcureSim benchmark card (skeleton; filled in M2–M5)

ProcureSim is a separately implemented procurement task environment with five simulated services. It shares the ActionProof codebase, so it is not an independent external environment.

## Transaction semantics
See `docs/stage2/TRANSACTION_SEMANTICS.md`.

## Fault taxonomy
Multi-axis (state observability, contract coverage, recoverability, agent observability); see `PLAN.md` §G1.

## Data generation
## Split construction (dev / validation / confirmatory) and leak checks
## Verifier assumptions
See `docs/stage2/VERIFIER_SPEC.md`.
## Metric definitions
See `docs/stage2/METRICS.md`.
## Known limitations
## Intended uses
Studying how reward information shapes evidence acquisition; testing execution-verification layers against write-side faults.
## Prohibited interpretations
Results are not evidence about real payment systems, real suppliers, or models outside the Aya family. Agreement between our verifiers is not ground truth.
