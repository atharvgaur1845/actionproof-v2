# RG1 addendum 2 (registered before any RG1 result was read): what the gate's results do and do not show

Written 2026-11-28 11:17 IST. An independent hostile code review (no blockers, no hidden-state leakage) found
that several registered outcomes are produced by the gate's design and the benchmark's labels rather than by the
agent. PREREGISTRATION_RG1.md is not edited; its verdicts stay as registered. This addendum fixes, before the results
are read, how they will be reported and adds descriptive decompositions (`scripts/rg1_addendum2.py`).

## Disclosures fixed in advance

1. **H1 (zero false completion) is largely guaranteed by construction.** In this simulator a consistent read is a
   perfect oracle of the verifier's step verdict, and the gate verifies every "completed" claim with one. H1 measures
   the gate plus a perfect read, not D. It will be reported that way.
2. **H2 and much of H3 are relabelling.** Unavailable records occur only in the unresolvable regime, where the label
   says hand-off is correct; the gate turns any other report there, or no report, into a hand-off. In the dev pilots
   about three quarters of the STR gain came from D's honest "failed" reports becoming hand-offs. STR, VTC and unsafe
   will be reported separately for resolvable and unresolvable tasks, next to reference agents behind the gate that
   make no call, claim completion at once, or report failure at once.
3. **The gate writes terminal reports.** When the agent stops without a report, the gate finalises (verified
   completion or hand-off); NG gets nothing. This step is in the hashed protocol (`RECOVERY_GATE_PROTOCOL.md` §1, §3)
   but not in the registration's rule list. Gate-written terminal reports will be counted separately. S3 (base, Aya
   Expanse 8B) will be described as system outcomes, mostly gate-written for weak agents.
4. **Completion removed on purpose.** The gate blocks same-key resends on unresolvable tasks (52 of 60 are keyed),
   which TXN1 addendum 3 shows settle the task. VTC on unresolvable tasks will be reported (expected to fall to 0).
5. **The "honest failure" asymmetry.** A report of "completed" with nothing written on an unresolvable task becomes a
   hand-off (STR credit); a report of "failed" with nothing written stays "failed". Reported via the references above.
6. **S2 no_retry ablation bug.** With the retry rule off, the receipt-repair rule treats an unread record as 0
   received and can force a duplicate receipt. That row overstates the retry rule's value; duplicate receipts in it
   will be counted and the row labelled accordingly. Confirmatory arms are unaffected.
7. **Errata (no effect on results):** the registration says 23 tests in test_gate.py (22 there, plus 1 Space-vendor
   test); addendum 1 says 60 of 60 transient tasks (80 of 80; 60 was a subset); gate.py's docstring cites a
   non-existent test_leakage.py; the freeze lists omit unchanged dependencies (integrity/sim/agents.py, truth.py,
   env.py; stage3/env.py), none changed since Nov 22.

## RG2 (same reviewer)

RG2's budget of 3 is the smallest that reads through rg2_test's 2-read outage, and SEL1 chose 3 alongside that
outage length. H1 and H2 are both driven by the t2 group. RG2 will be reported as a mechanism check ("a bounded
re-read fixes over-escalation on outages shorter than the budget"), not as evidence of general selectivity; an outage
of 3 reads would be handed off.
