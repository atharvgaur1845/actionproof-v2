# Spoken pitches

Three lengths of the same story. They follow the thesis of `README.md`, `docs/submission/WRITEUP.md` and
`docs/submission/FINAL_STORY_DECISION.md`. Each number is in `claims/claims.yaml` or the file named in
`docs/submission/JUDGE_QA.md`.

**Delivery rules.**
- Say "in a simulator".
- Never present one live run as a rate.
- Do not say "safe", "solved" or "beats every competitor".

---

## 30 seconds (about 80 words)

> A payment tool can say "ok" when nothing was paid, or time out when the payment went through. Agents that trust
> the reply get this wrong: on our locked benchmark of 440 tasks, trusting the reply falsely completes 86% of them.
> We trained a 3-billion-parameter Cohere model to check the record, repair and hand off. That raised verified
> completion from 0.51 to 0.82 on 450 frozen tasks. A hand-written script still does better. Everything is in a
> simulator, and every number has a file and a command.

## 60 seconds (about 160 words)

> When an agent pays a supplier, the reply does not prove what happened. The same "ok" can come back when nothing
> was stored, and a timeout can hide a payment that did go through.
>
> We built a locked benchmark of 440 such tasks in a procurement simulator, with a deterministic verifier that reads
> the final database. Trusting the reply falsely completes 86% of tasks.
>
> We trained Tiny Aya, a 3.35-billion-parameter Cohere model, with recovery demonstrations and then GRPO. On 450
> frozen tasks, verified completion rose from 0.51 to 0.82. That test was registered before any output, with three
> seeds. On the benchmark, the agent falsely completes 12% of tasks and leaves far fewer duplicates than retry with
> idempotency keys.
>
> What does not work, we report. A hand-written reconciliation script still beats the agent. Handing off to a human
> where that is the only safe action works in about four of ten episodes. And our own review found limits in the
> benchmark.
>
> RG1 (registered, sealed 440 tasks, D 3 seeds): behind the Recovery Gate, D's false completion falls from 0.112 to 0.000 and its verified completion on resolvable tasks rises from 0.824 to 0.892 (+0.068 [+0.046, +0.091]); unsafe episodes there fall from 482 to 34 of 3420. The hand-off result (1.000 against 0.011) is mostly the gate relabelling honest "failed" reports, and zero false completion is close to guaranteed by design (PREREGISTRATION_RG1_ADDENDUM2.md). B4 still completes more (1.000 on resolvable tasks).
>
> The live Space runs the real model, and two clean-room runs reproduced every number.

## 3 minutes (about 470 words)

**1. The problem (0:00–0:30).**
> A payment tool answers "ok". Did the payment happen? Not necessarily. The same request can get the same reply in
> four different realities: committed, never committed, timed out after committing, or timed out before. An agent
> that believes the reply reports success on a payment that does not exist. An agent that retries blindly pays
> twice.

**2. The measurement (0:30–1:00).**
> We built ProcureSim, an in-memory procurement simulator, and a locked benchmark on top of it: 440 tasks, eight
> fault classes, three evidence regimes and six out-of-distribution groups. A deterministic verifier scores the
> final database, never the agent's claim.
>
> Trusting the reply falsely completes 0.864 of tasks. Bounded retry with idempotency keys reaches 0.491 verified
> completion. Our hand-written reconciliation script reaches 0.864, with no false completion.

**3. What training changes (1:00–1:40).**
> Our policy is Tiny Aya Global, 3.35 billion parameters. Recovery demonstrations followed by GRPO raise sampled
> verified completion from 0.513 to 0.819 on 450 frozen tasks. The test was registered before any output, with three
> seeds per arm.
>
> On the benchmark, the trained agent falsely completes 0.121 of tasks, against 0.864 for trusting the reply. It
> leaves far fewer duplicate payments than retry with idempotency keys. The same recipe lifts four other Cohere
> bases, with one training seed each.
>
> Randomising the demonstrations raises the warm start on renamed tools from 0.137 to 0.741, and on an unseen
> workflow from 0.268 to 0.842. GRPO was not run on top, and the RL agent itself still fails renamed tools.

**4. What fails, and what that rules out (1:40–2:20).**
> A hand-written script still beats every agent we trained: 0.864 against 0.724. So we do not claim the agent
> replaces checks.
>
> Under unreadable evidence, the trained agent rarely hands off: 15 of 540 episodes. Our registered SEL1 experiment
> taught hand-off after repeated outage errors, at no cost on resolvable tasks. But where hand-off is the only safe
> action, it happens in 164 of 432 episodes, and one seed almost never does it.
>
> Our own hostile review found the benchmark weaker than designed. A policy that never reads but sends each keyed
> write twice reaches 308 of 440 verified completions, though with 132 false completions. Each finding is in a dated
> addendum; no registered verdict changed.
>
> RG1 (registered, sealed 440 tasks, D 3 seeds): behind the Recovery Gate, D's false completion falls from 0.112 to 0.000 and its verified completion on resolvable tasks rises from 0.824 to 0.892 (+0.068 [+0.046, +0.091]); unsafe episodes there fall from 482 to 34 of 3420. The hand-off result (1.000 against 0.011) is mostly the gate relabelling honest "failed" reports, and zero false completion is close to guaranteed by design (PREREGISTRATION_RG1_ADDENDUM2.md). B4 still completes more (1.000 on resolvable tasks).

**5. Why it matters, and how to check us (2:20–3:00).**
> The same pattern appears in accounts payable, refunds and payouts. A projection from our measured
> rates, with stated assumptions, puts trusting the reply at about ten times the trained agent's exposure. Scripted
> checks cut it further.
>
> Everything is public: the live Space runs the real model and reveals the database after each run; the replay site
> holds every episode; the models, traces and frozen splits are on Hugging Face. A smoke check of under a minute runs
> on any CPU, and one command recomputes every registered result from the released episodes. Two clean-room runs
> from a fresh clone reproduced them byte for byte.

---

## Answers to keep ready

The full list is in `docs/submission/JUDGE_QA.md`.
- **"Is it safe?"** No. It is a simulator result, and safe abstention is not solved.
- **"Did RL do it?"** Partly. The registered arms are not nested. Against its own SFT warm start, GRPO adds +0.140
  (exploratory). The reward's information source decides whether the agent reads back at all.
- **"Why not just the script?"** In this benchmark, use it. Our results show which checks matter. A trained agent is
  a second line of defence where the script does not reach.
