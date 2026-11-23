# The tool said success. The world disagreed.

Final research thesis. Written 2026-11-22 01:27 IST; branch `stage4`. Every number is in `docs/final/FINAL_CLAIMS.md`.

## The question

When a tool's acknowledgement cannot show whether a state-changing write actually happened, three questions follow:
- What does reinforcement learning teach a small agent to verify?
- Does the agent act correctly on the evidence it gets?
- Can training from faulted states improve its recovery?

## The setting

ProcureSim is a procurement back office. An agent orders goods, records receipts, pays suppliers and posts journal entries. Each write passes through one checkpoint, where we can make it store nothing, store part of the write, store it twice, or lose the reply. **The acknowledgement is byte-identical in every case.** The policy sees only tool replies. A state-only verifier sees the database. It was checked against a second verifier and golden traces, both written separately from the specification.

The model is Cohere Tiny Aya Global (3.35B) with LoRA. Training is GRPO through AgentTune, and every experiment was registered and hashed before it ran.

## Three findings

1. **What the reward sees changes what the agent checks.** Two rewards share the same binary form, warm start and trainer. The one that reads the database made read-back after a phantom payment the default: 1.00 in five of five seeds. The one that reads acknowledgements left the agent where it started: 0.00, identical to the untrained warm start. Verified completion rose by 0.27 to 0.32 per seed.
   - *Caveat:* the warm start already reads back when it samples (31% at temperature 1). Training selected a habit; it did not invent one. The registered follow-up ID1 confirms it: from a starting model that never reads, the same reward produced no reading at all in two seeds.
2. **Reading is not recovery.** Given the authoritative record, every model handles present-or-absent evidence correctly. Quantitative repair of a partial payment is different. A registered probe (E2) shows that one line reconciling the visible amounts raises correct refunds by +0.48, so the gap is acting on the evidence, not seeing it:
   - it is learned almost only under the state reward;
   - it is bound to workflow position: 0.99 to 1.00 when payment comes first, 0.00 after a goods receipt;
   - in recorded episodes, every trained model reads the partial payment after a goods receipt and still reports "completed".
3. **Training from the broken state helps, narrowly.** Recovery-start training is a known curriculum: the same GRPO and reward, with episodes that begin from a faulted write.
   - Against the state-reward agent: +0.21 verified completion in five of five seeds, and every run beats every run.
   - About half of that comes from one added fault type.
   - Against a control trained on the same faults, five runs per arm: completion from faulted states improves across training runs (p = 0.012). A partial-payment-specific gain is not established, and the three-seed safety cost reversed.
   - The gain survives Hindi and Hinglish requests and two-fault episodes. It vanishes under renamed tools and an unseen workflow, where agents replay the trained step sequence. Published work reports the same surface binding.

## What we do not claim

- No new optimiser or credit estimator. Our own estimator (FCA) failed its pre-set criterion and was dropped.
- No positive transfer: a sibling healthcare-administration domain got worse.
- No mechanism or circuit.
- Nothing about real deployments.

## One sentence

**Reliable agents must be judged by the state changes they actually produce, not the success messages they generate. Rewarding that state makes agents look, but looking is not yet knowing what to do.**
