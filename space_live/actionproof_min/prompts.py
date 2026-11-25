"""Prompts. Vendored verbatim from actionproof/bench/prompts.py (TOOL_RESULT_FORMAT only) and
actionproof/integrity/sim/prompts.py (PROC_NEUTRAL = PROC_SYSTEM)."""
from __future__ import annotations

from actionproof_min.env import PROC_SYSTEM

PROC_NEUTRAL = PROC_SYSTEM

TOOL_RESULT_FORMAT = "Tool result for {name}: {content}\nContinue the task. Call submit_report when you are finished."
