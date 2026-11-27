# TXN1 addendum 3: construct-validity findings from the final hostile review (post hoc)

Written 2026-11-27 and committed at 10:32 IST (c62f92f; an earlier draft stated 10:35, which was wrong), **after every TXN1 result had been seen** (registered analysis task 1739; the
descriptive reruns; the SEL1 results of task 1812). It is not a new hypothesis and changes no registered test, file
or verdict. It records four problems an independent hostile review found in the benchmark's construction
(`research_audit/FINAL_REDTEAM_REVIEW.md`, Reviewers 1 and 2), and the exploratory analysis written to size them:
`scripts/audit/txn_validity.py` (task 1840) → `results/audit/txn_validity.{json,md}`.

**The registered results stand as registered:**
- H1 (B0's false completion by reality): unchanged.
- H3: the family is not supported; both DMR tests pass and both URR tests fail.
- H5: descriptive.

What changes is how three of them may be described.

## Finding 1: the "unresolvable" regime is unresolvable by reading, not by acting

The regime makes the faulted step's read and the audit log unavailable for the whole episode, and labels escalation
correct (`docs/txn/TXN_SPEC.md` §4, `actionproof/txn/tasks.py`). But the simulator's idempotency store keeps the
documented contract: a retry with the same key never pays twice. A phantom stores no key, so a same-key resend
commits once; a committed write returns the original row. On a keyed step (order or pay), a same-key resend
therefore makes the final state correct in every hidden reality. The id it returns also reveals which reality held.

All 15 txn_test quadruples fault a keyed step (2 order, 13 pay), so all 60 unresolvable txn_test tasks can be
settled this way. A post-hoc policy that never reads and sends each keyed write twice with the same key (BD) scores
VTC 308 of 440, with 0 unsafe of 60 unresolvable tasks.

**Consequences:**
- On keyed steps, escalation is *a* safe disposition, not *the* correct one. "Escalation is the safe action there"
  (H5) is withdrawn as a description.
- The trained agents' verified completions on unresolvable tasks are mostly same-key resends (for example, 168 of
  SEL1 CTL's 173), not lucky guesses, as earlier text said.
- Only unkeyed steps (receive) are unresolvable in the sense the specification intended. txn_test has none; SEL1's
  sel1_test has 48 such tasks (SEL1 addendum 1).

## Finding 2: DMR counts duplicates the environment injects

DMR is "any DUPLICATE verdict in the final state", whoever caused it. Two fault classes inject a duplicate
themselves: `dup_delivery` (a second delivery under one key) and `concurrent` (a colleague's write). A policy that
never reads cannot repair them. On the 320 in-distribution tasks, B3's DMR is 61 episodes. Of these, 50 are those two
classes and 11 come from the other classes (11 of 270).

D's advantage survives the split, and it is in both parts:

| Duplicates (in-distribution tasks) | D, 3 seeds | B3 |
|---|---|---|
| Other classes (agent-caused) | 15 of 2430 episodes | 11 of 270 |
| Injected classes (D repairs most) | 54 of 450 | 50 of 50 |

The H3 headline is reworded to say both parts, and it stays inside a family that is not supported.

## Finding 3: a read-free policy comes close on VTC

Post-hoc reference policies on txn_test (one deterministic episode per task):

| Policy | VTC | FCR | DMR |
|---|---|---|---|
| FF fire-and-forget | 211 of 440 | 229 | 82 |
| BD blind same-key double send | 308 of 440 | 132 | 82 |

BD's VTC (0.700) is close to D seed 0's 0.724. Its false completions (0.300 against D's 0.121) and duplicates (0.186
against D's 0.052) are not. VTC alone overstates the headroom that needs reconciliation. TXN results are therefore
reported with FCR and DMR beside VTC, and with BD as a reference.

## Finding 4: compensations are under-checked

`refund_payment` does not refuse a second refund of the same payment. The pay verdict checks that exactly one
unrefunded payment remains, not that net money (payments minus refunds) equals the PO amount. An agent that refunds
the same payment repeatedly can therefore score VTC with a negative net balance.

Re-scoring every recorded episode with a net-money postcondition finds 3 credited-but-net-wrong episodes in total:
- 2 of 30,800 TXN model episodes (A seeds 0 and 2);
- 0 of 7,200 sel1_test model episodes;
- 1 of 36,000 P1 test episodes (B seed 2);
- none for D, Dw, Dw_aug, SEL or CTL.

No registered conclusion changes. The hole stays in TXN v1, because the benchmark is locked. It is listed for v2.

## Also recorded

- **The idempotency model.** It is not Stripe's in every respect:
  - the same key with different parameters returns the original result instead of an error;
  - keys never expire;
  - there is no in-flight conflict.
- **"Concurrent write".** It is a scripted injection before the agent's call, not an interleaving with version
  preconditions.
- **Partial payments.** Paying the remainder of a partial payment is scored DUPLICATE.
- **Scorer independence.** The 25 scoring mutants cover `actionproof/txn/*.py`, not the ProcureSim verdict code
  (`actionproof/integrity/sim/truth.py`). The reference verifier shares an author and a specification with the main
  one.
- **The environment.** ProcureSim's state is in-memory tables of rows, not SQLite as several documents said; corrected.

## v2 changes (future work; nothing here is run)

1. Unresolvable tasks only on unkeyed steps, or with key loss modelled explicitly.
2. A net-money postcondition, with refund-once enforced.
3. DMR reported by origin.
4. Parameter-mismatch idempotency errors, and key expiry.
5. Real interleavings with version preconditions.
6. Mutation testing of the ProcureSim verdict code.
