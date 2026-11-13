# ActionProof: what the reward can see decides whether a tool agent checks its writes

**Track 3 · AgentTune → AuditKIT · Tiny Aya Global 3.35B**

Tool agents report a write's outcome by repeating the tool's acknowledgement, but real tools acknowledge writes that never committed. In ActionProof, a SQLite support environment, a phantom write returns a **byte-identical** success acknowledgement; a deterministic verifier scores episodes from database state, not the transcript.

**Question.** With warm start, data, optimiser and reward form fixed, we change only what the reward's goal term can observe. What does GRPO keep?

**Before → after.** On untouched confirmatory splits (n = 400), GRPO with the state-grounded PCSTR-d reward raises verified-success completion (VSC) from **0.6425** to **0.8425** (both seeds); a tool-trusting reward stays at 0.635–0.645.

**Finding 1: the reward's information channel decides whether the agent checks.**

- Two rewards of identical form: the goal term reads the database or the acknowledgements. From one warm start, the database version raises sampled post-write refund verification from 0.25 to 0.71–0.72; the acknowledgement version drives it to 0.10 (two seeds each). Pooled VSC gain over a tool-trusting reward: **+19.25 pp** [15.4, 23.3].
- It replicates on a second warm start (one seed): greedy verification 1.00 vs 0.00, VSC 0.845 vs 0.615.

**Finding 2: what is selected is a script.** Retries follow reads, but stop at the training depth. On a write tool unseen in training, agents read back 73% of the time, see `credits: []`, never retry, and report "completed".

**Finding 3: GRPO keeps the winning SFT template whole, including unpaid steps.**

- In a held-out credit workflow, report-blind state-reward arms dropped the credit call (0/200, replicated at seed 1), re-issuing the refund.
- With 20 credit demonstrations added to SFT and RL unchanged, the call survives under the same reward (**200/200**, 113 committed) and under the acknowledgement reward (200/200), as in the warm start (one seed per arm; a second is queued).
- This is an instance of conditional-policy selection (arXiv 2607.03478) in an agent with side effects.
- Disclosed bug: keys shared across operations voided the earlier warm start's credit calls.

**Measure.** Untrained Cohere models below 32B score VSC ≈ 0; Aya Expanse 32B scores 0.515. LLM judges, including AuditKIT TaskCompletion, accept 76–100% of false completions.

**What we do not claim.**

- That RL teaches lying: the untrained model already falsely claims success 87–96% of the time.
- That verification is general: banking transfer fails, and checking stays tied to the trained tool.
- That any arm teaches honest failure: a seed-0 gap did not replicate (seed 1: 0.26 vs 0.24).
- That probes show the model "knows" it failed, or that the A5 environment shows learning (all policies at ceiling).

**Rigour.** Seven hashed pre-registrations; a deviation log records our own bugs. The registered confirmatory test passed: PCSTR-d − outcome = +1.5 pp [0.6, 2.6], p = 0.025 (underclaim avoidance). AuditKIT reproduces our VSC on 36/36 final files.

**ROI.** One H200 serves ≈ 9,600 tickets/hour (≈ $52 per 100k tickets).
