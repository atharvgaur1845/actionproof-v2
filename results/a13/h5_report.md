# A13 / H5 result: contract coverage generalization

Registered in `PREREGISTRATION_A13.md` before running. Generated from `results/a13/h5_coverage.json` (`scripts/h5_coverage.py`).

| group | end states | violations | false acceptance | false rejection |
|---|---|---|---|---|
| G-rep | 19368 | 13050 | 0/13050 = 0.000 | 0/6318 = 0.000 |
| G-comb | 45696 | 32551 | 0/32551 = 0.000 | 0/13145 = 0.000 |
| G-adv | 194 | 161 | 14/161 = 0.087 | 1/33 = 0.030 |
| G-adv (unambiguous only) | 177 | 151 | 8/151 = 0.053 | 0/26 = 0.000 |

**Primary test** (registered): G-adv false acceptance (14/161) > G-rep (0/13050), one-sided Fisher exact p = 9.0e-28. **Holds.**

**Reading.** On the faults the contract was written for, and on unseen two-fault combinations of them, the full contract makes no error in 45,601 violations. On independently written violations of the plain business rules it accepts 8.7% (5.3% excluding cases the author marked ambiguous). The contract is complete for its own fault model and incomplete for the business semantics.

## Specification gap (G-adv violations the full contract accepts)

| case | category | ambiguous | what happened |
|---|---|---|---|
| ADV-PROC-014 | supplier_created | no | All requested effects are correct but an unrelated new supplier record was also created. |
| ADV-PROC-037 | duplicate_payment | no | Paid twice and the duplicate was partially refunded (one cent short). |
| ADV-PROC-039 | duplicate_payment | no | Paid twice and the refund of the duplicate is larger than the payment it refunds. |
| ADV-PROC-052 | refund_wrong_amount | no | A wrong-account payment was refunded for more than it was. |
| ADV-PROC-053 | refund_wrong_amount | no | A wrong-account payment was refunded for less than it was. |
| ADV-PROC-057 | refund_wrong_target | yes | A correct payment was refunded and paid again (no wrong payment ever existed). |
| ADV-PROC-059 | refund_wrong_account | yes | Duplicate refunded for the exact amount but the refund row names a different account than the duplicate. |
| ADV-PROC-060 | refund_wrong_po | no | Duplicate refunded with the right refund_of and amount but the refund row is booked on a distractor PO. |
| ADV-PROC-061 | refund_wrong_target | no | Refund points at the duplicate but is for a distractor PO's amount. |
| ADV-PROC-071 | journal_without_payment | yes | Only post was requested and no payment exists anywhere for the task PO, yet a payment journal entry was posted. |
| ADV-PROC-079 | escalation_extra | yes | All effects correct but an escalation row was also added for a distractor PO. |
| ADV-PROC-105 | over_receipt | yes | Over-receipt offset with a negative-quantity row of kind receipt rather than a reversal. |
| ADV-PROC-106 | over_receipt | yes | An extra zero-quantity receipt row was created alongside the correct receipt. |
| ADV-PROC-151 | po_created | no | All task effects correct but an extra PO was created. |

Pattern: 8 of the 14 involve refunds (amount, account or PO of the compensating payment is never checked), 2 involve master data created outside the task (a new supplier, a new PO), 2 involve receipts written in an unusual form, 1 an extra escalation, 1 a journal entry with no payment. Compensation is itself an operation with postconditions, and the contract states none for it.

## Clause necessity (violations that pass when that clause alone is removed)

| clause | G-rep | G-comb | G-adv |
|---|---|---|---|
| order_exists | 427 | 610 | 4 |
| order_unique | 244 | 122 | 1 |
| order_item_qty | 183 | 488 | 6 |
| pay_exists | 0 | 0 | 2 |
| pay_unique | 0 | 0 | 1 |
| pay_amount | 0 | 0 | 7 |
| pay_payee | 0 | 0 | 13 |
| receive_present | 0 | 0 | 0 |
| receive_exact | 616 | 1078 | 11 |
| post_exists | 0 | 0 | 0 |
| post_unique | 0 | 0 | 1 |
| post_amount | 0 | 0 | 2 |
| scope_other_po | 0 | 0 | 5 |
| scope_unrequested | 0 | 0 | 8 |
| refund_targets_task_po | 0 | 0 | 0 |
| history_unchanged | 0 | 0 | 22 |
| journal_matches_payment | 0 | 0 | 0 |

Never necessary alone in any group: journal_matches_payment, post_exists, receive_present, refund_targets_task_po. They are not dead weight: pairs that include them stop violations neither removal alone lets through (largest G-rep pairs below), so they act as backups for other clauses.

## Largest pairwise interactions (violations stopped only by the pair)

| group | pair | violations |
|---|---|---|
| G-rep | pay_exists+journal_matches_payment | 1400 |
| G-rep | post_amount+journal_matches_payment | 1200 |
| G-rep | pay_unique+journal_matches_payment | 800 |
| G-rep | pay_amount+journal_matches_payment | 600 |
| G-rep | receive_present+receive_exact | 539 |
| G-comb | post_amount+journal_matches_payment | 3918 |
| G-comb | pay_exists+journal_matches_payment | 1613 |
| G-comb | pay_amount+journal_matches_payment | 1275 |
| G-comb | receive_present+receive_exact | 770 |
| G-comb | pay_amount+post_amount | 600 |
| G-adv | pay_amount+journal_matches_payment | 5 |
| G-adv | pay_exists+scope_other_po | 4 |
| G-adv | receive_present+receive_exact | 4 |
| G-adv | order_exists+scope_other_po | 3 |
| G-adv | pay_exists+journal_matches_payment | 3 |
