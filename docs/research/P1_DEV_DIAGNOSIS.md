# P1 development diagnosis (descriptive; written before any P1 test result was read)

Written 2026-11-23, about 23:30 IST, from `results/p1/dev/*` (arms A, B, C; D was still training). The development split selects only the demonstration agent (`PREREGISTRATION_P1.md` §7). Every arm runs the frozen test regardless, and nothing registered changes because of this note.

## Development sampled VSC (mean of three seeds; files `results/p1/dev/<arm>_s<k>__<split>.summary.json`)

| Arm | p1_dev_full | p1_dev_rec |
|---|---|---|
| A, recovery-start GRPO | 0.758 | 0.705 |
| B, A + reconciliation line | 0.757 | 0.681 |
| C, A + recovery SFT | 0.842 | 0.861 |

## Why B is below A on recovery starts

**Short-payment recovery starts** (`p1_dev_rec`, seed 0, greedy, 14 tasks):

| | A | B | C |
|---|---|---|---|
| Verified success | 13 / 14 | 2 / 14 | 14 / 14 |
| False success | 1 | 12 | 0 |

Sampled VSC on the same tasks: A 0.339, B 0.161, C 0.804.

On seed 0, greedy decoding, `p1_dev_rec`, every task B fails and A passes (11 of 11) is a short payment (`partial` on `pay_supplier`). B passes no task that A fails.

**Mechanism** (task `p1_dev_rec-0014`):
1. The line reads "purchase order total 802370 cents; live payments 601777 cents; outstanding 200593 cents".
2. The agent pays exactly the outstanding 200593 cents, with the original idempotency key.
3. The payment service deduplicates that call and returns the original payment id. No money moves.
4. The agent posts the journal entry for the short amount and reports "completed": a false success.

**Reading.** The line exposes the shortfall, but its "outstanding" figure suggests a top-up. The registered policy ("pay once", §2) rejects that repair. E2's +0.48 was measured on first repairs after a goods receipt, a different decision point.

**Product lesson.** A runtime reconciliation hint must state the shortfall in terms of the allowed repair, or be paired with a gateway rule. A bare "outstanding" amount steers a policy toward the forbidden action.

This is a hypothesis from development data. It is not tested here, and B is evaluated on the test exactly as registered.

## Note on C

Greedy C scores the same on every seed (0.889 and 0.922). Sampled scores differ by under 0.01 across seeds. Recovery SFT moves the three RSC seeds to nearly the same policy.
