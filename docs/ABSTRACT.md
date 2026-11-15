# ActionProof: what the reward can see decides whether a tool agent checks its writes

**Track 3 · AgentTune → AuditKIT · Tiny Aya Global 3.35B**

Real tools acknowledge writes that never committed. In ActionProof, a SQLite support environment, phantom writes return a **byte-identical** success acknowledgement; a deterministic verifier scores episodes from database state, not the transcript.

**Question.** When acknowledgements can contradict the database, what does GRPO select if its goal reward reads the acknowledgement instead of the state?

**Before → after.** On untouched pre-registered confirmatory splits (n = 400, both seeds), GRPO with the state-grounded PCSTR-d reward raises verified-success completion (VSC) from **0.6425** to **0.8425**, **+20.25 pp** [16.4, 24.3] over a tool-trusting reward. On a stricter metric that also forbids unrequested writes, the state arms are unchanged and the gap is +41.9 pp.

**Finding 1: the reward's information source decides whether the agent checks.** Two rewards of identical form from one warm start; only the goal term's source differs. Test-IID VSC 0.825 vs 0.605 (two seeds each); sampled refund read-back after a success acknowledgement 0.25 → 0.71–0.72 vs 0.10. On a later warm start, greedy read-back is 1.00 vs 0.00, two seeds per arm.

**Finding 2: task composition explains why the acknowledgement reward stops checking.** On refund timeouts and unrecoverable refunds, acting on a read forfeits success acknowledgements, so the acknowledgement reward pays readers less. Pre-registered: removing those two classes restores read-back to **0.669**; removing 60 random tasks leaves **0.000**; the state reward reads back 1.000 on both pools. Channel × pool interaction **+0.678** [0.605, 0.750] (one seed per cell; task-level CI).

**Finding 3: execution evidence in a judge's prompt is not reliably used.** AuditKIT TaskCompletion (Aya Expanse 32B) accepts **95%** of false completions. Given the database's own receipt ledger, it still accepts **72%** of the failures the ledger exposes (our prompt: 83% → 46%; Aya 8B: 96–100%). The deterministic verifier rejects all of them.

**Limits we found ourselves.**

- GRPO selects a script kept whole from SFT: retries stop at the training depth; on an unseen write tool agents read, see nothing, and report "completed".
- In a corrected environment (v2) where timeouts can fail before the commit, the state-reward agents read the order, then report done without retrying (PCSTR-d 0.18 vs 0.71 for the tool-trusting agent). They learned the timeouts they were trained on.
- Our report-blind state reward pays continuing after an unrecoverable refund (+0.167). A dependency-aware variant fixes the ordering offline; no policy was trained on it.

**What we do not claim.** That RL teaches lying (the warm start already claims false success 87–96% of the time). That verification is general (banking transfer fails). That the result holds beyond one model family, one environment, or one seed per Amendment 8 cell.

**Rigour.**

- Eleven hash-frozen registrations; a public deviation log.
- A reference verifier written without our code agrees on 4,000 / 4,000 confirmatory episodes.
- Registered confirmatory test passed: PCSTR-d − outcome +1.5 pp [0.6, 2.6], p = 0.025.

**ROI.** Single-request p50 is 2.5 s per episode for PCSTR-d vs 10.2 s for the tool-trusting agent (A100).
