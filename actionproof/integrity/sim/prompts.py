"""ProcureSim system prompts. NEUTRAL trains and evaluates every model; VERIFY is the prompt-only baseline."""
from __future__ import annotations

from actionproof.integrity.sim.env import PROC_SYSTEM

PROC_NEUTRAL = PROC_SYSTEM

PROC_VERIFY = PROC_SYSTEM + """

Reliability rules:
- Tool acknowledgements can be wrong. After every write, confirm it with the matching read tool (find_supplier_order, list_payments, list_goods_receipts, get_journal_entries).
- If a write times out, check the matching read tool before retrying; when retrying, reuse the same idempotency_key.
- Only report "completed" after you have confirmed every requested change; otherwise report "failed"."""

PROC_PROMPTS = {"neutral": PROC_NEUTRAL, "verify": PROC_VERIFY}
