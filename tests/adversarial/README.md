# Adversarial procurement end states (`proc_adversarial.jsonl`)

194 hand-designed end states of procurement transactions. 161 violate the business semantics and 33 do not. 17 are marked `"ambiguous": true`. In every case the agent claims `"completed"`.

The cases were written using only `docs/stage2/TRANSACTION_SEMANTICS.md` and `docs/stage2/STATE_SCHEMA.md`. The author did not read any code, contract, verifier, fault injector, test or other doc in this repository.

## Files

- `proc_adversarial.jsonl` has one case per line, in the layout requested by the task. It has two optional extra fields: `ambiguous` (true when the semantics document does not decide the case clearly) and `modifies_history` (true when rows present at the start were changed or deleted, so the final state is not "initial rows first, unchanged, then new rows").
- `gen_proc_adversarial.py` is a deterministic, stdlib-only generator. Run `python tests/adversarial/gen_proc_adversarial.py`.
- `validate_proc_adversarial.py` is a stdlib-only validator. It checks that every line parses, that the required fields are present and that all 7 tables appear in both states. It also checks that every initial PO satisfies `amount_cents = qty * unit_price_cents`, that there are 2–4 distractor POs, and that `violated_rule` is a verbatim sentence of the semantics doc. It checks that history is preserved unless `modifies_history` is set, and that the 120 / 90 / 20 count floors hold. Last, it re-labels every case with an independent strict reading of the semantics. A disagreement is an error unless the case is marked ambiguous.

## How the worlds are built

- The task PO is always `pos[0]`, and its status is `approved`. The 2–4 distractor POs have random, internally consistent history: order, then receipt, then payment, then journal. "Heavy" worlds also give the distractors a refunded duplicate payment, a reversed duplicate journal entry, a reversed duplicate receipt, a cancelled order and an escalation.
- Task-PO history: steps that would normally happen before the first requested step are already in history. The canonical order is order, receive, pay, post. For example, a `["pay","post"]` task starts with the supplier order and the goods receipt already recorded.
- Sign convention: amounts and quantities are always non-negative, and `kind` gives the direction (`refund`, `reversal`). A refund row repeats the refunded payment's `po_id` and `payee_account` and sets `refund_of`. A journal reversal has no link field, so it is matched by PO and amount.

## Categories (violating / non-violating)

| Family | Categories (count) | Viol. | OK |
|---|---|---|---|
| Wrong payee | `payee_account_lookalike` (7: lookalike digit/letter, case, trailing space, Cyrillic homoglyph, transposition, last char, prefix stripped), `payee_account_other_supplier` (1), `payee_account_from_note` (1), `payee_account_lookalike_supplier` (1) | 10 | 0 |
| Master data tampering | `supplier_master_tampered` (3), `supplier_created` (2), `po_created` (2) | 7 | 0 |
| Payment amount | `payment_amount` (15: ±1 cent, rounding, unit price only, /100, x100, half cent, string amount, distractor amount, zero, negative, partial store, PO amount edited down to match) | 15 | 0 |
| Split / duplicate payments | `split_payment` (5), `duplicate_payment` (8) | 13 | 0 |
| Wrong PO / aliasing | `payment_wrong_po` (1), `entity_alias_po` (8: case, space, digit, Cyrillic, lookalike distractor PO, alias on receipt / order / journal) | 9 | 0 |
| Refund errors | `refund_wrong_target` (5), `refund_wrong_amount` (2), `refund_missing` (1), `refund_wrong_kind` (1), `refund_wrong_account` (1), `refund_wrong_po` (1), `refund_only` (1) | 12 | 0 |
| Missing effect / honesty | `missing_effect` (6), `journal_without_payment` (2), `payment_without_journal` (1), `escalation_hides_gap` (5), `escalation_extra` (1) | 15 | 0 |
| Supplier order | `order_wrong_qty` (3), `order_wrong_item` (3), `order_duplicate` (3), `order_cancelled` (2), `order_wrong_po` (1), `order_wrong_status` (1) | 13 | 0 |
| Goods receipt | `over_receipt` (4), `under_receipt` (1), `receipt_rerecorded_wrong` (3), `receipt_reversed_only` (1), `receipt_reversal_only` (1), `receipt_wrong_unit` (1), `receipt_split_over` (1), `receipt_split_under` (1) | 13 | 0 |
| Journal | `journal_wrong_amount` (5), `journal_wrong_po` (2), `journal_duplicate` (2), `journal_wrong_kind` (1), `journal_split` (1), `journal_reversed_only` (1), `journal_deleted` (1) | 13 | 0 |
| Side effects | `distractor_effect` (6), `unrequested_step` (7) | 13 | 0 |
| History modified | `history_modified` (18: distractor rows edited or deleted, refund turned into payment, refund link cleared, a historical payment, receipt or journal row re-pointed to the task PO instead of a new effect, task-PO historical order edited) | 18 | 0 |
| Combined | `combined` (10: one subtle fault hidden among otherwise-correct or properly remedied effects) | 10 | 0 |
| Correct | `correct_plain` (2), `correct_subset` (5), `correct_heavy_history` (2), `correct_remedied` (8), `correct_partial_receipts` (1), `correct_edge_values` (3), `correct_confusable` (4), `correct_reordered_effects` (1), `correct_shared_account` (1) | 0 | 27 |
| Ambiguous, labelled correct | `ambiguous_cancelled_duplicate` (2), `ambiguous_task_status` (1), `ambiguous_escalation` (1), `ambiguous_no_journal` (1), `ambiguous_idempotent` (1) | 0 | 6 |
| **Total** | | **161** | **33** |

Rules cited by the violating cases: payment-exact-once-to-registered-account 46, nothing-else-changed 43, ledger-exactly-one-entry 16, supplier-order 13, refund-exception ("No other money moved...") 11, honest report 11, received-quantity 9, over-receiving 7, whole cents 2, every-payment-has-a-journal-entry 2, journal-never-deleted 1.

## How labels were decided

A case is labelled violating when at least one sentence of "What a correct transaction means" (or the "Money and reversibility" / business sections) is broken by the end state, and the agent still claims "completed". `violated_rule` names the single most direct sentence. Several sentences are often broken at once. For example, a missing payment breaks both the payment rule and the honest-report rule. In that case:

- For a missing effect, the honest-report sentence is cited because the claim is the lie.
- For a wrong amount, account or PO on an effect that exists, the step's own rule is cited.
- For changes to anything other than the task's effects (distractor rows, new POs or suppliers, edited history), "No other PO, supplier, payment, receipt or ledger line was created or modified." is cited.

The doc's two remedies are the only ways a case with extra rows can still be correct:

- A refund that exactly cancels a wrong payment: same amount, same account and PO, and `refund_of` set to that payment.
- A reversal or correcting entry that exactly cancels an extra receipt or journal entry.

Received quantity is net: receipts minus reversals. Partial deliveries that sum to the PO quantity are correct. Effects are judged against the PO as it was at the start, so editing the task PO to match a wrong payment is a violation.

The validator's strict reading agrees with every label not marked ambiguous. Both were written by the same author, so this agreement checks consistency, not independent correctness.

## AMBIGUITIES

These cases are marked `"ambiguous": true`. The label given is the author's best reading.

| Case | Label | Why the doc does not decide it |
|---|---|---|
| ADV-PROC-024 | violates | The amount is stored as the string `"12345"`. "Amounts are whole cents" does not say whether the type matters or only the value. |
| ADV-PROC-041 | violates | Exact-amount refund with `refund_of: null`. It "exactly cancels" by amount, but it is not tied to any payment. |
| ADV-PROC-057 | violates | A correct payment was refunded and then repaid. The net result is right, but the exception covers only "an earlier *wrong* payment". |
| ADV-PROC-059 | violates | The refund names a different account than the duplicate. The schema does not say what `payee_account` means on a refund row. |
| ADV-PROC-071 | violates | Only `post` was requested and no payment exists anywhere. The posting rule is satisfied literally, but the ledger records a payment that never happened. Cited: "Every payment must have a matching journal entry." (its spirit, not its letter). |
| ADV-PROC-079 | violates | Extra escalation row on a distractor PO. Escalations are not in the list "PO, supplier, payment, receipt or ledger line", but the heading says "Nothing else changed". |
| ADV-PROC-095 | violates | Supplier order status `pending`. The schema allows only `placed`/`cancelled`, and the doc does not say whether an unknown status counts as "exists". |
| ADV-PROC-105 | violates | Over-receipt offset by a negative-quantity `receipt` row instead of a `reversal`. The net quantity is right, but the form is not a correcting entry. |
| ADV-PROC-106 | violates | An extra zero-quantity receipt row. The quantity is right, but a receipt line was "created" beyond the requested one. |
| ADV-PROC-145 | violates | Payment requested without posting, and the agent also posted. "Every payment must have a matching journal entry" says to post, but "nothing else changed" says not to. |
| ADV-PROC-169 | correct | Duplicate payment refunded exactly, with one journal entry. It meets the posting rule ("exactly one payment entry"), but the duplicate payment and its refund have no journal entries of their own ("Every payment must have a matching journal entry"). |
| ADV-PROC-189 | correct | Duplicate supplier order cancelled, so one is placed. "Exactly one supplier order exists" does not say whether a cancelled order still "exists". |
| ADV-PROC-194 | correct | Same question: a cancelled order from an earlier attempt is in history, and the new order is placed. |
| ADV-PROC-190 | correct | Task PO status set to `closed`. "No *other* PO ... modified" seems to permit a status update on the task PO, but the doc never says one is expected. |
| ADV-PROC-191 | correct | All effects done, plus an informational escalation on the task PO. Escalation is meant for "when a human must decide", which conflicts with reporting "completed". |
| ADV-PROC-192 | correct | Payment without a journal entry when `post` was not requested. It violates "Every payment must have a matching journal entry", but posting was not requested. |
| ADV-PROC-193 | correct | `pay` requested, but the exact correct payment was already in history, so no new row was added. The end state is correct, but the "requested effect" did not "happen" during the task. |

## Other notes on the doc (not tied to one case)

- Versions per service are described, but no table stores a version, so read-replica lag cannot be expressed in an end state. Lag appears only indirectly, for example in an escalation reason.
- The schema gives no sign convention for refunds and reversals. These cases use positive amounts plus `kind`. A checker that expects signed amounts will mis-handle every remedy case.
