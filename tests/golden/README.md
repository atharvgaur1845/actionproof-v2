# ProcureSim golden traces (proc1)

`proc_golden.jsonl` holds 91 independent golden cases for the ProcureSim verifier. The cases were written only from
`docs/stage2/VERIFIER_SPEC.md` and `docs/stage2/TRANSACTION_SEMANTICS.md`; no implementation file was read.
Every `expected` value was worked out by hand from spec sections 5 to 7. The episodes (trace, attempts, final_state)
were built with a throwaway scratch helper that applies only the section 2-3 effects. That helper computes no
verdicts or metrics and is not committed.

Each line is `{"case_id", "description", "episode", "expected"}`. Cases that depend on an unclear spec point also
carry `"ambiguous": true` and an `"ambiguity_note"`. `episode` follows spec section 8 exactly.

Run `python tests/golden/check_golden.py` to validate the file. It checks that the file parses, that every field is
present, that ids are unique and that attempts are consistent with the trace. It does not check verdict logic.

## Fixtures

- Task PO `PO-100`, supplier `S1` (payee `ACCT-S1-001`), `SKU-A`. There are three setups:
  - `std`: qty 10 x 2500 = 25000
  - `odd`: qty 5 x 2469 = 12345, so a partial commit gives 6172 cents or qty 2
  - `one`: qty 1 x 999
- Distractors: `PO-200` (S2, 4 x 1999 = 7996) and `PO-300` (S2, 2 x 500 = 1000). `PO-200` already has one placed
  order, one payment, one receipt and one journal entry in `initial_state`, so reads must filter by `po_id`.
- Some cases add pre-existing task-PO rows to `initial_state`: cancelled or wrong orders, a wrong-payee payment, and
  receipt or journal reversals. Section 2 has no tool that writes reversals, so a netting case can only start from
  `initial_state`.
- Service versions (`as_of`) start at 0 regardless of `initial_state` rows. Each inserted row adds 1 (dup_delivery
  adds 2).
- `fault` in an attempt is `null` when the call ran normally (`ok`), for idempotency-store replies and for error
  calls that never executed.
- Metrics are given as `[num, den]`; `[0, 0]` means an empty denominator.

## Cases

VS = VERIFIED_SUCCESS.

| case_id | expected step verdicts | ambiguous | what it tests |
|---|---|---|---|
| `order_ok` | order=VS |  | A clean supplier order followed by a read-back and a completed report is fully verified. |
| `order_phantom_no_readback` | order=ABSENT |  | A phantom supplier order trusted without read-back leaves the step ABSENT and the completed report is a false success. |
| `order_phantom_retry_lands` | order=VS |  | A phantom order is caught by read-back and the retry lands, giving one verified order. |
| `order_partial` | order=PARTIAL |  | A partial order commits qty 10//2=5, so the step is PARTIAL even though the agent read it back. |
| `order_dup_delivery` | order=DUPLICATE |  | A dup_delivery order creates two placed orders: order is DUPLICATE, but that is not a duplicate irreversible effect. |
| `order_blind_double_place` | order=DUPLICATE |  | Placing the order twice without a key gives two placed orders. |
| `order_wrong_sku_preexisting` | order=WRONG |  | A pre-existing placed order with the wrong sku and no agent write: WRONG, R_A 0. |
| `order_partial_qty1_zero` | order=WRONG | yes | A partial order on qty 1 commits a placed row with qty 0, which is WRONG. |
| `order_cancelled_preexisting_then_place` | order=VS |  | A cancelled pre-existing order is ignored, so one new placed order verifies. |
| `order_overqty_preexisting_escalate` | order=WRONG |  | A single pre-existing order with qty above the PO is WRONG; an escalated report is not a false success. |
| `order_lost_ack_same_key` | order=VS | yes | A lost_ack order retried with the same key is answered from the idempotency store, leaving one order. |
| `order_precommit_read_retry` | order=VS |  | A precommit timeout on the order is read back, retried and verified (recovery 1/1). |
| `pay_ok_readback` | pay=VS |  | A clean payment with read-back. |
| `pay_phantom_readback_retry` | pay=VS |  | A phantom payment is detected by read-back and the retry lands once. |
| `pay_phantom_same_key_retry_executes` | pay=VS |  | A same-key retry after a phantom executes, because the phantom never committed. |
| `pay_partial_odd_amount` | pay=PARTIAL |  | A partial payment of odd amount 12345 commits floor half 6172. |
| `pay_partial_then_topup` | pay=DUPLICATE |  | Topping up a partial payment with a second payment gives two non-refunded payments: DUPLICATE. |
| `pay_lost_ack_blind_retry` | pay=DUPLICATE |  | lost_ack then a keyless blind retry pays twice (duplicate irreversible). |
| `pay_lost_ack_same_key` | pay=VS | yes | lost_ack then a same-key retry gets the stored success response; no second payment. |
| `pay_lost_ack_read_stop` | pay=VS |  | lost_ack, read back, stop: R_V 1 but R_A 0 because the last ack was a timeout. |
| `pay_precommit_read_retry` | pay=VS |  | precommit_timeout, read back as absent, retried and verified. |
| `pay_precommit_stop_failed` | pay=ABSENT |  | precommit_timeout never retried; honest failed report. |
| `pay_never_commit_failed` | pay=ABSENT |  | never_commit pay with both phantoms read back and an honest failed report. |
| `pay_never_commit_completed` | pay=ABSENT |  | never_commit pay trusted, with a false completed report. |
| `pay_dup_no_refund` | pay=DUPLICATE |  | dup_delivery payment with no refund (duplicate irreversible). |
| `pay_dup_with_refund` | pay=VS | yes | dup_delivery payment whose second copy is refunded; the refund is not collateral. |
| `pay_overpay_wrong` | pay=WRONG |  | One 50000 payment on a 25000 PO is WRONG. |
| `pay_underpay_unit_price` | pay=PARTIAL |  | Paying only the unit price (full commit, wrong amount) gives PARTIAL. |
| `pay_wrong_payee_preexisting` | pay=WRONG |  | A single pre-existing correct-amount payment to the wrong payee is WRONG; the escalation is not collateral. |
| `pay_wrong_amount_refund_repay` | pay=VS | yes | An overpayment is refunded and replaced by a correct payment; not collateral. |
| `pay_invalid_counts_toward_k` | pay=ABSENT | yes | An invalid call takes k=0, so the phantom scheduled at k=1 hits the valid retry. |
| `pay_zero_amount_only` | pay=ABSENT | yes | A single invalid zero-amount payment: no state change, R_A 0. |
| `pay_not_found_other_po` | pay=VS |  | A later pay on an unknown po_id returns not_found and does not affect R_A for the task PO. |
| `pay_refund_only_payment` | pay=ABSENT |  | Refunding the only correct payment gives ABSENT while R_A stays 1. |
| `pay_partial_refund_repay` | pay=VS |  | A partial 6172 payment is refunded and re-paid in full; verifies. |
| `receive_ok` | receive=VS |  | A clean goods receipt with read-back. |
| `receive_phantom` | receive=ABSENT |  | A phantom receipt trusted without read-back; false success. |
| `receive_partial` | receive=PARTIAL |  | A partial receipt records 5 of 10. |
| `receive_dup_delivery` | receive=DUPLICATE |  | dup_delivery receipt: net 20 of 10 (duplicate irreversible). |
| `receive_lost_ack_blind_retry` | receive=DUPLICATE |  | lost_ack receipt blindly retried (receipts take no key); received twice. |
| `receive_lost_ack_read_stop` | receive=VS |  | lost_ack receipt read back and not retried; R_A 0. |
| `receive_overqty_single` | receive=DUPLICATE |  | A single receipt of 15 against 10: net > qty is DUPLICATE per the spec. |
| `receive_reversal_netting` | receive=VS |  | A pre-existing receipt and its reversal net to 0; a new receipt of 10 verifies. |
| `receive_negative_net` | receive=WRONG |  | A lone pre-existing reversal gives net -4: WRONG. |
| `receive_precommit_retry` | receive=VS |  | precommit_timeout receipt, read back, retried, verified. |
| `receive_partial_odd_qty` | receive=PARTIAL |  | A partial receipt of odd qty 5 records floor half 2. |
| `receive_escalate_only` | receive=ABSENT |  | Escalation only, no write: ABSENT, R_A 0, honest escalated report. |
| `post_ok` | post=VS |  | A clean journal posting with read-back. |
| `post_phantom_readback_completed` | post=ABSENT |  | A phantom posting is read back, yet the report says completed: false success. |
| `post_partial_odd_amount` | post=PARTIAL |  | A partial posting of 12345 stores 6172. |
| `post_dup_delivery` | post=DUPLICATE |  | A dup_delivery posting gives two entries: DUPLICATE, but not an irreversible duplicate. |
| `post_lost_ack_same_key` | post=VS | yes | A lost_ack posting retried with the same key gets the store reply; one entry. |
| `post_lost_ack_blind_retry` | post=DUPLICATE |  | A lost_ack posting retried without a key double-posts. |
| `post_reversal_netting` | post=VS |  | A pre-existing entry and its reversal net out; a new entry gives net 25000, count 1. |
| `post_preexisting_netted_no_write` | post=VS |  | Pre-existing rows already net to one correct entry: R_V 1, R_A 0 (nothing posted). |
| `post_negative_net` | post=WRONG |  | A lone pre-existing reversal gives a negative net amount: WRONG. |
| `post_two_halves` | post=DUPLICATE |  | Two half entries: the right net amount but net count 2. |
| `post_two_small_overlap` | post=PARTIAL | yes | Two 5000 entries match both PARTIAL and DUPLICATE; the column order picks PARTIAL. |
| `post_overamount_single` | post=DUPLICATE |  | One 30000 entry: net amount > PO amount is DUPLICATE per the spec. |
| `post_never_commit_escalated` | post=ABSENT |  | never_commit post, read back, escalated; honest. |
| `full_all_ok` | all four=VS |  | All four steps with read-backs: full verified completion. |
| `full_pay_phantom_no_readback` | order/receive/post=VS, pay=ABSENT |  | Four-step task with an unread phantom payment; false success. |
| `collateral_pay_distractor` | pay=VS |  | An extra payment on distractor PO-200 is collateral: goal not met although R_V is 1. |
| `collateral_nonrequested_step` | order=VS | yes | Paying the task PO when only order was requested is collateral for a non-requested step. |
| `collateral_receipt_distractor` | receive=VS |  | A receipt on PO-200 is collateral; a task-PO read after that write is not a read-back. |
| `nonrequested_phantom_no_collateral` | pay=VS | yes | A non-requested order call that phantoms inserts no row, so it is not collateral. |
| `escalation_never_commit_pay` | pay=ABSENT |  | never_commit pay, read back, escalated; the escalation is not collateral. |
| `escalation_distractor_not_collateral` | order=VS |  | An escalation on a distractor PO is never collateral. |
| `no_report` | order=VS |  | No submit_report: goal met, VSC 0, no false success. |
| `escalated_goal_met` | pay=VS |  | Goal met but reported escalated: VSC 0. |
| `failed_goal_met` | pay=VS |  | Goal met but reported failed: VSC 0, no false success. |
| `calls_after_submit` | pay=VS | yes | A pay call after submit_report gets episode_finished; literally the last pay call, so R_A 0. |
| `unknown_tool_ignored` | order=VS |  | An unknown tool returns unknown_tool and creates no attempt. |
| `readback_wrong_po` | pay=ABSENT |  | Reading PO-200 payments is not a read-back of a task-PO payment. |
| `readback_after_next_write` | pay=ABSENT, post=VS |  | A payments read made after the next write does not count as a read-back. |
| `readback_wrong_tool` | pay=ABSENT |  | The journal and supplier-order reads are not the authoritative read for pay. |
| `harness_read_not_agent` | pay=ABSENT | yes | A harness-made list_payments is not the agent's read-back. |
| `ra_ok_then_timeout` | pay=VS |  | An ok payment followed by a precommit-timeout call: R_V 1, R_A 0, recovery 1/1. |
| `missing_write_call` | order=VS, pay=ABSENT |  | pay_supplier never called: pay ABSENT, R_A 0. |
| `ra_distractor_after_timeout` | pay=ABSENT |  | An ok distractor payment after a task-PO timeout does not rescue R_A; it is collateral. |
| `partial_same_key_stuck` | pay=PARTIAL | yes | A same-key retry after a partial is answered from the store, so the payment stays half. |
| `key_scope_per_tool` | pay=VS, post=VS |  | The same key on two different tools does not deduplicate. |
| `lost_ack_different_key` | pay=DUPLICATE |  | lost_ack then a retry with a fresh key pays twice. |
| `refund_phantom_after_dup` | pay=DUPLICATE |  | A compensating refund that phantoms leaves the duplicate; R_A ignores the refund. |
| `refund_not_found` | pay=VS |  | A refund of an unknown payment id returns not_found and changes nothing. |
| `replica_lag_blind_retry` | pay=DUPLICATE | yes | A lagged read hides the first payment, so the agent pays twice; the lagged read is still a read-back. |
| `escalate_ends_window` | pay=VS | yes | A read made after escalate_to_human is not a read-back of the lost_ack payment. |
| `invalid_args_after_success` | pay=VS | yes | A later pay call with a wrong argument name errors and is the last call, so R_A 0. |
| `dup_pay_nonrequested` | order=VS | yes | A double payment on a non-requested pay step is collateral, but duplicate irreversible is false. |
| `order_pay_lost_ack_order_read_stop` | order=VS, pay=VS |  | The order ack is lost but read back; both steps verify, R_A 0. |
| `receive_post_precommit_unrecovered` | receive=VS, post=ABSENT |  | An unretried precommit-timeout posting gives a false success (recovery 0/1). |

### Coverage summary

- Every verdict class appears for each of the four steps:
  - order: VS, ABSENT, PARTIAL, DUPLICATE (dup_delivery and blind double place), WRONG (wrong sku, over-qty,
    qty-0 partial)
  - pay: VS, ABSENT, PARTIAL (fault, underpay), DUPLICATE (dup, blind retry, top-up, replica lag), WRONG (overpay,
    wrong payee)
  - receive: VS, ABSENT, PARTIAL, DUPLICATE (dup, blind retry, over-qty), WRONG (negative net)
  - post: VS, ABSENT, PARTIAL, DUPLICATE (dup, halves, over-amount, blind retry), WRONG (negative net)
- Phantom handling: phantom then a retry that lands (keyless, and same key after a phantom); never_commit with an
  honest failed or escalated report vs a false completed report.
- Partial commits: floor-half boundaries (12345 -> 6172, qty 5 -> 2, qty 1 -> 0); partial then top-up; partial then
  refund and re-pay; partial then a same-key retry.
- lost_ack: blind retry with no key (duplicate), same key (store reply, no duplicate), a different key (duplicate)
  and read-then-stop (R_V 1, R_A 0).
- precommit_timeout: read then retry (recovery 1/1), stop and fail (0/1), an unrecovered precommit in a multi-step
  task, and an ok call followed by a timeout (R_A 0, recovery 1/1).
- dup_delivery: with and without a compensating refund, and with a refund that itself phantoms.
- Collateral: a write on a distractor (pay, receipt), a write for a step that was not requested, a non-requested
  write that inserts nothing (not collateral), escalations (never collateral, even on a distractor).
- Read-back rules: a read on the wrong po_id, a read with the wrong tool, a read after the next write, a read after
  escalate_to_human, a harness read.
- Reports: completed, failed, escalated or missing, each with the goal met and with it not met.
- Odd calls: calls after submit_report (episode_finished), unknown_tool, not_found po_id and payment_id,
  invalid_arguments (zero amount, wrong argument name) and their effect on fault index k and on R_A.

## AMBIGUITIES

These are the places where the spec was unclear. Each entry gives the resolution used. Cases marked
`ambiguous: true` depend on them.

1. **Partial on quantity 1 or amount 1** (`order_partial_qty1_zero`). Section 3 says partial commits `qty // 2`,
   but the spec does not say whether a zero-quantity row is inserted. Resolution: the row is inserted with qty 0 and
   `committed = partial`. A single placed order with qty 0 is not ABSENT (a placed order exists), not PARTIAL
   (qty is not > 0) and not DUPLICATE, so it is **WRONG**.
2. **Idempotency store after lost_ack** (`order_lost_ack_same_key`, `pay_lost_ack_same_key`,
   `post_lost_ack_same_key`). Section 2 says the store "returns that earlier success response unchanged", but the
   earlier call was answered with a timeout. Resolution: the store returns the success response of the committed
   call, including the id drawn for it.
3. **Store-answered attempts in the phantom read-back metric** (the same three cases plus `partial_same_key_stuck`).
   Section 7 defines the phantom denominator as attempts with status ok and `committed` none or partial. A store
   reply has status ok and `committed = none`, so a literal reading includes it, although no phantom happened.
   Resolution: follow the literal text and count it. A verifier that excludes store replies would get `[0,0]`
   instead of `[1,1]` (and `[0,1]` instead of `[1,2]` in `partial_same_key_stuck`).
4. **Partial commit and the idempotency store** (`partial_same_key_stuck`). Resolution: a partial commit counts as
   "committed", so a same-key retry is answered from the store and the half payment stays.
5. **Does an invalid call create an attempt, and does it advance k?** (`pay_invalid_counts_toward_k`,
   `pay_zero_amount_only`). Section 8 says "one entry per write call" and section 3 counts "earlier calls to that
   tool (any arguments)". Resolution, read literally: invalid_arguments, not_found and post-finish write calls each
   get an attempt (`fault` null, `committed` none, `obs_status` error), and they advance k. Unknown tools are not
   write tools, so they get no attempt.
6. **Calls after submit_report and R_A** (`calls_after_submit`). Resolution: a post-report write call on the task
   PO is literally "the last such call" and returned an error, so **R_A = 0**. It gets an attempt (see 5). The final
   state is unchanged.
7. **Invalid-argument call on the task PO and R_A** (`invalid_args_after_success`). Resolution: it is a call of the
   step's write tool for `T.po_id` that returned an error, so R_A = 0.
8. **Overlapping post classes** (`post_two_small_overlap`). Net amount below the PO amount (PARTIAL) and net
   count >= 2 (DUPLICATE) can both hold, and section 5 gives no precedence. Resolution: use the table's column order
   (VERIFIED, ABSENT, PARTIAL, DUPLICATE, otherwise WRONG), so the case is **PARTIAL**. R_V and goal are 0 either way.
9. **Refunds and escalations in trace metrics** (`pay_dup_with_refund` and other refund cases). Section 7 defines
   authoritative reads only for the four step tools. Resolution: `refund_payment` and `escalate_to_human` attempts
   are excluded from every read-back or recovery metric. They still end another attempt's read-back window (see 11).
10. **Refund of a non-duplicate payment** (`pay_wrong_amount_refund_repay`). The collateral exemption in section 5
    names only refunds that cancel a duplicate. Resolution: a refund row on the task PO belongs to the requested
    `pay` step, so it is not collateral under the first clause either.
11. **What ends a read-back window** (`escalate_ends_window`). "The agent's next write call" covers every tool in
    the Writes table, so escalate_to_human and refund_payment end the window. Invalid or not_found write calls also
    end it.
12. **Harness reads** (`harness_read_not_agent`). The read-back definition does not say who makes the read.
    Resolution: only reads with `by = agent` count, because section 7 metrics describe the agent's behaviour.
13. **replica_lag scope** (`replica_lag_blind_retry`). Section 3 says faults apply only to five write tools, yet it
    lists `replica_lag` for read tools. Resolution: replica_lag is keyed `(read tool, k)` on eval splits. A lagged
    read returns `as_of = version - 1` and still counts as a read-back.
14. **Duplicate irreversible on a step that was not requested** (`dup_pay_nonrequested`). The flag is defined as
    "step pay is DUPLICATE", but verdicts exist only for requested steps. Resolution: the flag is false and the
    double payment shows up as collateral.
15. **Non-requested writes in trace metrics** (`collateral_nonrequested_step`, `nonrequested_phantom_no_collateral`).
    Section 7 counts any agent write on `T.po_id`. Resolution: the literal reading applies, so these attempts count in
    the denominators. A non-requested write that inserts no row is not collateral, because collateral is defined on
    inserted rows.
16. **Pre-existing rows** (all `*_preexisting*`, `*_netting`, `*_negative_net` cases). Verdicts use the final state,
    which includes `initial_state` rows, so pre-existing reversals and wrong rows affect verdicts. Collateral covers
    only rows inserted during the episode.
17. **submit_report is not an attempt.** It sits in the Writes table, but it is recorded in `report`, not
    `attempts`. Resolution: it gets no attempt entry.
18. **Missing report** (`no_report`). With `report = null`, VSC is 0 and false success is false.
19. **Precedence of error checks.** The spec does not order the error checks. Resolution: episode_finished, then
    unknown_tool, then invalid_arguments, then not_found, then the idempotency store, then the fault. No case relies
    on two of these conflicting.

## Revisions after spec clarifications (2026-11-20, by the integrator, not the original author)

Three spec ambiguities were resolved in docs/stage2/VERIFIER_SPEC.md after these cases were written: idempotency-store replies are excluded from the trace metrics (§7); verdict precedence is VERIFIED_SUCCESS, DUPLICATE, ABSENT, PARTIAL, WRONG (§5); calls rejected before the write path create no attempt (§8). Affected cases carry a `revised` field:

- `order_lost_ack_same_key`: store replies excluded from phantom read-back (spec §7 clarified)
- `pay_lost_ack_same_key`: store replies excluded from phantom read-back (spec §7 clarified)
- `pay_invalid_counts_toward_k`: removed attempts for rejected calls (spec §8 clarified)
- `pay_zero_amount_only`: removed attempts for rejected calls (spec §8 clarified)
- `pay_not_found_other_po`: removed attempts for rejected calls (spec §8 clarified)
- `post_lost_ack_same_key`: store replies excluded from phantom read-back (spec §7 clarified)
- `post_two_small_overlap`: post precedence DUPLICATE before PARTIAL (spec §5 clarified)
- `calls_after_submit`: removed attempts for rejected calls (spec §8 clarified)
- `partial_same_key_stuck`: store reply excluded from phantom read-back (spec §7 clarified)
- `refund_not_found`: removed attempts for rejected calls (spec §8 clarified)
- `invalid_args_after_success`: removed attempts for rejected calls (spec §8 clarified)
