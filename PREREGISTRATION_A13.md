# Amendment 13: contract coverage generalization (H5)

Registered 2026-11-20 (commit time of this file and `results/a13_hashes.json`), before `scripts/h5_coverage.py` was run on any group (only case counts were computed while building it). Plan: `docs/stage2/PLAN.md` §G4.

## Question
What does an explicit transaction contract establish, what does it miss when a clause is removed or when clauses interact, and which violations can no clause express?

## Instrument
`actionproof/integrity/clauses.py`: 17 named clauses (order exists, unique, item and quantity; payment exists, unique, amount, payee; receipt present, exact; journal exists, unique, amount, matches the payment; no new rows on other POs; no new rows for steps that were not requested; refunds only of task-PO payments; history unchanged). The full set agrees with the ProcureSim truth verifier on whether the goal was met for 3,080 of 3,080 scripted episodes (checked while building; reported). A transaction is accepted only if every enabled clause holds.

## Groups (labels never come from the clause contract)
- **G-rep**: the 200 `proc_dev` worlds x every requested step x one fault kind (phantom, partial, lost_ack, precommit_timeout, dup_delivery, never_commit) x the six scripted defences of `scripts/h7_matrix.py`; label = the truth verifier's goal-met verdict.
- **G-comb**: the same worlds with two faults on two different requested steps (kinds phantom, partial, lost_ack, precommit_timeout; none of these pairs occurs in any training split); same defences and labels.
- **G-adv**: `tests/adversarial/proc_adversarial.jsonl`, 194 end states (161 violating, 33 correct, 17 marked ambiguous), written by an independent author from `docs/stage2/TRANSACTION_SEMANTICS.md` and `docs/stage2/STATE_SCHEMA.md` only; label = the author's `violates_semantics`. Reported with and without the ambiguous cases.

## Measures
False acceptance (accepting a violation) and false rejection (rejecting a correct transaction) of the full contract per group; clause necessity (violations that pass when that clause alone is removed) with the top families; redundancy (clauses with no necessity in any group); pairwise interactions (violations stopped only by a pair of clauses together); the specification gap (G-adv violations the full contract accepts), listed case by case.

## Primary test
Full-contract false acceptance on G-adv is greater than on G-rep (one-sided Fisher exact test, alpha 0.05). Falsified if it is not. Everything else is descriptive.

## Expected (stated before running)
G-rep false acceptance near 0 (the clauses were written against the simulator's own faults); G-adv false acceptance above 0, concentrated in categories no clause covers (for example master-data tampering, string or negative amounts handled loosely, aliasing that lands on the task PO). H14 (residual false commitment equals the leak union) is not a claim; it may follow algebraically from shared definitions.

## Frozen files
`PREREGISTRATION_A13.md`, `actionproof/integrity/clauses.py`, `scripts/h5_coverage.py`, `scripts/h7_matrix.py`, `tests/adversarial/proc_adversarial.jsonl`, `tests/adversarial/README.md`, `data/bench/proc/proc_dev.jsonl`.
