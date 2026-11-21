# S3 addendum 3 withdrawn before any training (2026-11-22 01:29)

`PREREGISTRATION_S3_ADDENDUM3.md` (commit `198bea0`) is kept unchanged and hash-frozen. It is **withdrawn**: no state-only run was trained, evaluated or started. Its tasks (970–976) were removed from the queue before any of them was claimed, and `runs/` holds no `s3_stateonly_*` directory.

**Why.** The red-team review (`docs/RESEARCH_RED_TEAM_REVIEW.md`) found the design degenerate:
- **The idempotency store hides the history.** With the history hidden, the agent re-issues the write with the same idempotency key the hidden prefix used. It does so in 295–299 of 300 episodes, because the key convention is `<step>-<po_id>`. The simulator then replays the stored acknowledgement, so the agent sees exactly what a full-episode agent sees after its own faulted write. For partial payments, the "state-only" arm is observationally the class-matched control, so C1 was null by construction.
- **Receive-then-pay starts become traps.** For these starts (30% of recovery starts), the agent's natural unkeyed receipt doubles the quantity, so R_V is 0 whatever the agent does about the payment.

A replacement design was not registered in its place. The review's alternative (own-history starts plus acknowledgement-only starts) is assessed in `docs/final/EXPERIMENT_STATUS.md`.
