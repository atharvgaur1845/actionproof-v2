# ActionProof: agents learn to look, not to use what they see

**Track 3, Agentic Fine-tuning · AgentTune → AuditKIT (+ CuratorKIT) · Cohere Tiny Aya Global and Aya Expanse 8B / 32B**

Tool-using agents report a write's result by repeating the tool's acknowledgement. Real tools lie: writes are acknowledged but never committed, committed partially, or time out after committing. ActionProof is a SQLite customer-support environment where a phantom write returns a **byte-identical** success acknowledgement. A receipt ledger and a deterministic verifier score every episode from database state, never from the transcript.

**Question.** Hold the warm start, data, optimiser and reward form fixed, and change only *what the reward can see*. Which behaviours does GRPO keep, and which does it remove?

**Finding 1: what the reward observes decides whether the agent checks.**
- Two rewards have identical form: one computes its goal term from the database, the other from the tool's acknowledgements.
- From the same warm start, the state-grounded reward raises sampled post-write refund verification from 0.25 to 0.71–0.72.
- The acknowledgement-grounded reward drives it down to 0.10.
- Two training seeds each; the effect replicates on a second warm start (verification 1.00 vs 0.00).
- Pooled over test-IID and held-out faults, PCSTR-d beats a tool-trusting reward by **+19.25 pp VSC** [15.4, 23.3], and binary `outcome` beats it by +18.0 pp [13.9, 22.3].
- An LLM-judge reward was **saturated** (mean 0.99, within-batch standard deviation ≈ 0.03), so its arm's 0.61 is not evidence that the judge taught anything.

**Finding 2: what gets selected is a script, not a skill.**
- Trained agents read back the database after the write, and on a write tool never seen in training they still read back 73% of the time.
- They see the contradiction there (`credits: []`), never retry, and report "completed".
- Every retry follows a read (evidence-gated), but the number of retries is capped at the training depth. Unrecoverable faults are never reported as failures: 0/150 for every model and every middleware.
- In a later round, all state-reward arms stopped calling the unfamiliar tool. They re-issued the refund instead and wrote summaries claiming the credit was applied (200/200; one seed). Acquiring evidence transferred; using it did not.

**Finding 3 (one seed, provisional): an honesty term can switch honesty off.**
- We added demonstrations so the warm start sometimes reports honest failure.
- A report-blind state reward then produced honest failure on 30% of unrecoverable tasks (45/50 refund tasks), with no premature giving-up on recoverable twins.
- The same reward plus nondisclosure and underclaim penalties produced **0%**: it retried four times and claimed success.
- Seed 1 and an ND-only ablation are running.

**Measure.**
- Untrained Cohere models below 32B score VSC ≈ 0. Aya Expanse 32B reaches 0.515, and prompting it to "verify every write" lowers that to 0.33.
- LLM judges, including AuditKIT's TaskCompletion on Aya 32B, accept 76–100% of false completions.

**What we do not claim.**
- That RL teaches lying: the untrained model already claims false success 87–96% of the time.
- That verification is a general skill: banking transfer fails.
- That probes show the model "knows" it failed: a text baseline matches them.
- That the proof-carrying A5 environment shows learning: every policy, including prompt-only, is at ceiling.

**Rigour.**
- Six pre-registrations and amendments with hashed splits and scoring code, plus a deviation log that includes our own bugs.
- A one-shot confirmatory test on untouched splits.
- Every number traces to a file in `results/`. Models, traces and logs are on Hugging Face.

**ROI.** One H200 serves about 9,600 tickets per hour: about $52 per 100k tickets self-hosted, versus $228–684 at list API prices on the measured workload.
