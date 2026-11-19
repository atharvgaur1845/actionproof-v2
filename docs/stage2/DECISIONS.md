# Stage 2 decision log

Newest last. Times are IST. Each entry says what was decided, who decided it, and why.

| When | Decision | By | Why |
|---|---|---|---|
| 2026-11-19 22:40 | The Stage-2 deadline is Nov 25 (Nov 24 23:59 AoE = Nov 25 17:29 IST). | user | The official page has two dates; the user confirmed the later one. |
| 2026-11-19 22:40 | No official rubric exists, so the plan infers judging criteria from the public page. | user | Final judging is on a video or slides; criteria are not published. |
| 2026-11-19 22:40 | Core models stay in the Aya family. Other models only as teacher, validator or judge. | user | Competition rule. |
| 2026-11-19 22:40 | HF credits may be spent up to $200 with a running log and a hard cap. | user | $200 Stage-2 credits; each spend is logged in `HF_CREDITS.md`. |
| 2026-11-19 22:45 | Planning agents run on Opus 5.5 instead of Fable. | user | Fable 5.1 returned "requires usage credits". This is a logged deviation from the earlier rule that Fable plans. |
| 2026-11-20 00:40 | Research first: the procurement GRPO replication and the core experiments come before the product. | user | The single-environment training result is the largest scientific limitation. |
| 2026-11-20 00:40 | One human annotator (the user), 60 episodes. | user | Time available. A second annotator is added if one becomes available. |
| 2026-11-20 00:40 | Video: the user narrates and records; Claude writes the script and slides and drives the demo commands. | user | |
| 2026-11-20 01:30 | Plan revision 2 after an external review (7.8/10): three-layer framing, three paired seeds, frozen fault classes, four gateway modes, verifier validation, precise guarantees. | user (review) | See `PLAN.md` Context. |
| 2026-11-20 02:00 | Plan revision 3 after a second review (8.4/10): reward-truth table and phantom-success read-back as the primary endpoint; protected confirmatory splits; multi-axis fault taxonomy with three estimands; seeds 3 minimum, 5 target; contract-coverage generalization with independent adversarial faults; research and product framed separately. Plan approved. | user | See `PLAN.md` Context. |
| 2026-11-20 01:55 | One H200 worker job holding 2 GPUs runs two lanes (`slurm/worker2.sh`) instead of two 1-GPU jobs. | Claude | `qos_gpu_h200` allows one submitted job per user (MaxSubmitPU = 1) but 2 GPUs. `slurm/worker.sh` now writes a lane-specific owner id so the lanes never re-queue each other's tasks. |
| 2026-11-20 02:15 | If a Slurm job has waited more than 3–4 hours, its work moves to HF Jobs (within the HF caps, logged in `HF_CREDITS.md`). | user | Tight deadline; queue time is the main risk. |
| 2026-11-20 02:30 | Seeds run as separate queued tasks across every pool (H200 ×2 lanes, A100 ×2, H100 ×2 workers) and HF Jobs. HF runs seed 1 first, then only seeds no Slurm worker has started. A task is claimed by an atomic move out of `queue/<pool>/pending/`, into `running/` (Slurm) or `queue/hf/` (HF), so no seed is paid for twice. | user | Queues are contended (no start estimate); do not wait, do not waste money. |
