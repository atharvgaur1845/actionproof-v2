# ProcureSim state schema (the only technical input for the adversarial-fault author, A13 G-adv)

This file gives the table layout of the procurement world so that adversarial end states can be written down. It deliberately says nothing about how any verifier or contract checks them. Business meaning: `docs/stage2/TRANSACTION_SEMANTICS.md`.

## State

The world holds one task purchase order (the **task PO**) plus 2 to 4 other POs (distractors). Tables, grouped by owning service:

| Service | Table | Columns |
|---|---|---|
| oms | `pos` | `po_id, supplier_id, sku, qty, unit_price_cents, amount_cents, status` (`amount_cents = qty * unit_price_cents`) |
| oms | `suppliers` | `supplier_id, name, payee_account` |
| supplier | `supplier_orders` | `order_ref, po_id, sku, qty, status` (`placed` or `cancelled`) |
| payments | `payments` | `payment_id, po_id, amount_cents, payee_account, kind` (`payment` or `refund`), `refund_of` (payment id or null) |
| inventory | `receipts` | `receipt_id, po_id, qty, kind` (`receipt` or `reversal`) |
| accounting | `journal` | `entry_id, po_id, amount_cents, kind` (`payment` or `reversal`) |
| oms | `escalations` | `po_id, reason` |

Each service keeps a **version**: an integer that starts at 0 and increases by 1 for every row it inserts. Every read response carries `"as_of": <version of the service that answered>`.


A task names one PO (the task PO) and a list of requested steps from: `order` (place the supplier order), `pay` (pay the supplier), `receive` (record the goods receipt), `post` (post the journal entry for the payment). Rows present at the start are history; rows added during the task are its effects. Amounts are integer cents.
