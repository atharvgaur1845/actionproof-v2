## Credit calls on `e_heldout_ctrl` by task fault (episodes calling `apply_store_credit` / n)

| fault (tool / type) | init_a6 | a6_pcstr_d_s0 | a6_pcstr_d_noreport_s0 | a6_pcstr_d_noND_s0 | a6_outcome_s0 | a6_ack_outcome_s0 |
|---|---|---|---|---|---|---|
| add_audit_note / F1_phantom | 29/30 | 0/30 | 0/30 | 0/30 | 0/30 | 28/30 |
| add_audit_note / F4_timeout | 21/22 | 0/22 | 0/22 | 0/22 | 0/22 | 19/22 |
| close_ticket / F1_phantom | 25/25 | 0/25 | 0/25 | 0/25 | 0/25 | 25/25 |
| close_ticket / F4_timeout | 26/26 | 0/26 | 0/26 | 0/26 | 0/26 | 26/26 |
| issue_refund / F1_phantom | 24/24 | 4/24 | 4/24 | 3/24 | 3/24 | 23/24 |
| issue_refund / F4_timeout | 22/23 | 23/23 | 23/23 | 22/23 | 23/23 | 20/23 |
| none / - | 48/50 | 0/50 | 0/50 | 0/50 | 0/50 | 46/50 |

## The call immediately before the first credit call

- **init_a6**: issue_refund → success ×194, query_ledger → ok ×1
- **a6_pcstr_d_s0**: issue_refund → success ×14, reconcile_order → success ×9, query_ledger → ok ×4
- **a6_pcstr_d_noreport_s0**: issue_refund → success ×18, reconcile_order → success ×5, query_ledger → ok ×4
- **a6_pcstr_d_noND_s0**: issue_refund → success ×14, reconcile_order → success ×8, query_ledger → ok ×3
- **a6_outcome_s0**: reconcile_order → success ×21, query_ledger → ok ×3, issue_refund → success ×2
- **a6_ack_outcome_s0**: issue_refund → success ×186, issue_refund → error ×1

State arms (4): 25 tasks in which every state arm calls the credit tool; 27 in which any does.
