# ActionProof: agents learn to look, not to use what they see

**Track 3, Agentic Fine-tuning · AgentTune → AuditKIT (+ CuratorKIT) · Cohere Tiny Aya Global and Aya Expanse 8B / 32B**

Tool-using agents report a write's result by repeating the tool's acknowledgement. Real tools lie: writes are acknowledged but never committed, committed partially, or time out after committing. ActionProof is a SQLite customer-support environment where a phantom write returns a **byte-identical** success acknowledgement. A receipt ledger and a deterministic verifier score every episode from database state, never from the transcript.

**Question.** Hold the warm start, data, optimiser and reward form fixed, and change only *what the reward can see*. Which behaviours does GRPO keep, and which does it remove?

**Finding 1: what the reward observes decides whether the agent checks.**
- Two rewards have identical form: one computes its goal term from the database, the other from the tool's acknowledgements.
- From the same warm start, the state-grounded reward raises sampled post-write refund verification from 0.25 to 0.71–0.72.
- The acknowledgement-grounded reward drives it down to 0.10.
- The ack arm has two seeds (0.102 / 0.100); greedy VSC has two seeds for every arm. The effect replicates on a second warm start (greedy verification 1.00 vs 0.00).
- Pooled over test-IID and held-out faults, PCSTR-d beats a tool-trusting reward by **+19.25 pp VSC** [15.4, 23.3], and binary `outcome` beats it by +18.0 pp [13.9, 22.3].
- An LLM-judge reward was **saturated** (mean 0.99, within-batch standard deviation ≈ 0.03), so its arm's 0.61 is not evidence that the judge taught anything.

**Finding 2: what gets selected is a script, not a skill.**
- Trained agents read back the database after the write, and on a write tool never seen in training they still read back 73% of the time.
- They see the contradiction there (`credits: []`), never retry, and report "completed".
- Every retry follows a read (evidence-gated), but the number of retries is capped at the training depth. Unrecoverable faults are never reported as failures: 0/150 for every model and every middleware.
- In a later round, all state-reward arms stopped calling the unfamiliar tool. They re-issued the refund instead and wrote summaries claiming the credit was applied (200/200; one seed). Disclosure: an environment bug (idempotency keys shared across operations) had already made the warm start's credit calls silently void, so the baseline never worked either. Acquiring evidence transferred; using it did not.

**Finding 3: what GRPO keeps is the warm start's template, including steps the reward never pays for.**
- We added 20 demonstrations of the credit workflow with the credit's own idempotency key to the warm start. The credit tool appears in SFT only: no training task contains it, so RL never rewards it.
- The same report-blind state reward that had removed the credit step (0/200) now keeps it: **200/200** non-colliding credit calls on the held-out workflow, 113 committed, identical to its warm start. The acknowledgement-grounded arm also keeps it (200/200), while still dropping verification (test-IID 0.62 vs 0.85 VSC). Coverage of a step the training tasks never contain is set by SFT, not by either reward (one seed each; seed 1 pending).
- On fixed, identical prefixes (exact probabilities, no sampling), the state-reward arms check the ledger after a refund acknowledgement with probability 0.54–0.65, against 0.16 for the warm start and 0.06 for the ack arm. Finding 1 holds at the token level.

**Measure.**
- Untrained Cohere models below 32B score VSC ≈ 0. Aya Expanse 32B reaches 0.515, and prompting it to "verify every write" lowers that to 0.33.
- LLM judges, including AuditKIT's TaskCompletion on Aya 32B, accept 76–100% of false completions.

**What we do not claim.**
- That RL teaches lying: the untrained model already claims false success 87–96% of the time.
- That verification is a general skill: banking transfer fails.
- That probes show the model "knows" it failed: a text baseline matches them.
- That the proof-carrying A5 environment shows learning: every policy, including prompt-only, is at ceiling.
- That any reward arm teaches or suppresses honest failure. A seed-0 gap between arms (P(failed | same evidence) 0.71 vs 0.11) did not replicate at seed 1 (0.26 vs 0.24), and in the training prompt every model reports "failed" at that point with probability ≥ 0.99. The rate depends on one extra tool in the prompt, not on the reward.

**Rigour.**
- Seven pre-registrations and amendments with hashed splits and scoring code, plus a deviation log that includes our own bugs.
- A one-shot confirmatory test on untouched splits passed as registered: PCSTR-d beats binary `outcome` by +1.5 pp [0.6, 2.6] (p = 0.025); the gain is underclaim avoidance.
- Every number traces to a file in `results/`. Models, traces and logs are on Hugging Face.

**ROI.** One H200 serves about 9,600 tickets per hour: about $52 per 100k tickets self-hosted, versus $228–684 at list API prices on the measured workload.
