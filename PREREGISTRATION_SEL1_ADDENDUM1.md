# SEL1 addendum 1: what the registered tests measure (post hoc)

Written 2026-11-27, 10:40 IST, **after the registered analysis (task 1812) and every SEL1 result had been seen**. It
changes no registered test, file or verdict: SEL1 is supported as registered (H1, H3, H4), and H2 passes. It records
what an independent hostile review found about the tests' construct validity (`research_audit/FINAL_REDTEAM_REVIEW.md`,
Reviewers 1 and 2; `PREREGISTRATION_TXN1_ADDENDUM3.md`), sized by the exploratory analysis
`scripts/audit/txn_validity.py` (task 1840; `results/audit/txn_validity.json`, key `unresolvable`).

## 1. H1's tasks could be settled without escalating

All 60 unresolvable txn_test tasks fault a keyed step (order or pay). A same-key resend settles them (TXN1 addendum 3,
finding 1). Pooled episodes, 3 seeds:

| txn_test unresolvable (540) | escalated | verified completion | unsafe | other (mostly an honest "failed" report) |
|---|---|---|---|---|
| SEL | 525 | 1 | 7 | 7 |
| CTL | 1 | 173 (168 after a same-key resend) | 12 | 354 |
| D (TXN1) | 15 | 32 | 11 | 482 |

So H1's +0.970 mostly measures replacing an honest "failed" report with a call to `escalate_to_human`. Unsafe actions
fall from 12 to 7 of 540. CTL's completions were same-key resends, not lucky guesses as earlier text said.

## 2. Selectivity is shown against a 2-read transient outage only

- `t2` is a 2-read outage. B5, the demonstrated rule, escalates after 3 unavailable reads
  (`actionproof/sel1/policy.py`, `READS_BEFORE_ESCALATING = 3`). `t2` therefore tests imitation of that counter
  inside the demonstrated budget. It is not a held-out transient length beyond it.
- `audit_down` cannot fire. B5 reads the audit log only after its primary reads fail. On `audit_down` they succeed,
  so a B5 imitator never observes the outage: SEL's reads and calls there match `std`. The registered falsifier
  ("a shortcut 'audit log down → escalate' fails here") was never exercised. 0 escalations of 360 on `audit_down`
  is not evidence of selectivity.

## 3. Where the outcome is truly unresolvable, the effect is partial and seed-fragile

Only the unkeyed receive step is unresolvable by any action. sel1_test has 48 such tasks (`u_receive`, plus the
receive steps of `u_ood_wf`); 432 episodes over 3 seeds:

| sel1_test, unkeyed unresolvable (432) | escalated | unsafe |
|---|---|---|
| SEL | 164 (by seed: 55, 106, 3 of 144) | 215 |
| CTL | 0 | 293 |

## 4. The trigger is an explicit error

The escalation trigger is an explicit `service_unavailable` reply on the faulted step's read. It is not inferred
from the identical acknowledgements. The hidden reality does not change the correct decision on these tasks.
`research_audit/NOVELTY_POSITIONING_FINAL.md`'s claim that "the outage is hidden behind identical replies" is
withdrawn.

## How SEL1 is described from now on

Registered demonstrations of a scripted rule taught a 3.35B agent to call for a human after repeated
`service_unavailable` reads of the faulted step, including on held-out workflows. They did so without escalating on
a 2-read outage, and at no cost on resolvable tasks. That is what the supported tests show.

They do not show selective hand-off where it is uniquely correct. On the locked test, every such task could have been
settled by a same-key resend. Where nothing settles the outcome (the receive step), the arm hands off in 164 of 432
episodes, seed-fragile, and remains unsafe in 215.

**Missing controls, for any follow-up:**
- an escalate-on-first-error demonstration arm;
- an instruction-only arm;
- transient outages longer than the demonstrated budget;
- an audit-log outage the policy actually observes;
- unresolvability signalled other than by an error string.
