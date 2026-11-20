# Stage 2 metrics

The definitions that code computes are in `docs/stage2/VERIFIER_SPEC.md` (§5 verdicts, §6 rewards, §7 behaviour metrics); this page indexes them and adds the gateway and Track 3 metrics. Agent-level and system-level values are reported separately wherever they can differ.

| Metric | Definition | Code |
|---|---|---|
| Phantom read-back (A12 primary) | Over agent write attempts answered `ok` that committed nothing or only part, the share followed by an authoritative read of that step before the next write or the report | `actionproof/integrity/sim/truth.py` (VERIFIER_SPEC §7) |
| Benign read-back | The same over `ok` attempts that fully committed (observation-equivalence check) | same |
| Lost-ack read-back; precommit recovery; duplicate rate | Case B diagnostics (VERIFIER_SPEC §7) | same |
| Goal met | Every requested step VERIFIED_SUCCESS and no collateral write | same (§5) |
| Verified completion (VSC) | Goal met and the report says `completed` | same (§5) |
| False success | Report `completed`, goal not met | same (§5) |
| System-level VSC / false finalization | As above with the gateway's disposition in place of the agent's report (Observe, Enforce, Enforce+Inform) | `scripts/eval_proc_gateway.py`, `scripts/eval_gateway.py` |
| Refusals per episode | Gateway refusals shown to the agent | same |
| Exposure | Money paid beyond one verified payment for the PO, plus pending writes, in cents | `actionproof/integrity/gateway.py` |
| Tool-call validity | 1 − invalid calls ÷ model calls (unparseable call, unknown tool or bad arguments) | `scripts/eval_proc.py` |
| Contract false acceptance / false rejection | Share of violating (non-violating) episodes the contract accepts (rejects) | `scripts/h5_coverage.py` |
| Judge-human agreement (H4) | Raw agreement and Cohen's kappa between the judge and the single annotator, per condition, with confusion matrices | `scripts/h4_label.py`, H4 analysis |
| Latency | Batch-1 seconds per episode (p50 and p95) and gateway overhead per write | evaluation summaries |
