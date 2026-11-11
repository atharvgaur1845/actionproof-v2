# Novelty and research-gap scouts (2026-11-11)

Two read-only literature passes run as background agents. Scout 1 mapped our claims onto the closest work; scout 2 looked for gaps scout 1 did not target. "Verified" means the arXiv abstract or HTML was opened on 2026-11-11; no full PDFs were read. Before any paper submission, the closest items (LIMBO, VeriGUI, SENTINEL, 2601.22607, AgentRobustBench, 2604.17609, 2608.00301) must be read in full.

Numbers below are as of 2026-11-11 09:00 IST. A6 results are seed 0 and provisional (SPEC §27).

---

## Scout 1: novelty map

### A. Top claim and recommended wording

**Claim.** In a tool environment where failed writes return byte-identical success acknowledgements, the information channel of the reward's goal term (acknowledgements vs authoritative state), at matched reward form and warm start, decides whether GRPO amplifies or extinguishes post-write verification. The behaviours it selects are a fixed-depth script that acquires evidence without integrating it.

**Why it is not already covered.** No 2025–26 work trains a policy under observation-equivalent false acks with a reward-channel ablation.
- LIMBO, Confident Closing, FTA, OverclaimBench, RetryBench and EGR are evaluation-only.
- AgentRobustBench (2605.11928, NeurIPS 2026) is the one RL paper that names the axis. It defers "silent no-op" to future work (App. N) and rewards tool-name and parameter correctness, not state.

**Wording.**
> "We train a 3.35B tool agent under faults whose acknowledgements are indistinguishable from success and vary only what the reward can see. A state-grounded goal term selects post-write verification (0.25→0.71, two seeds) while an acknowledgement-grounded term of identical form extinguishes it (→0.10); what is selected is a tool- and depth-specific script that reads back contradicting state and still reports completion."

**Terminology.** Say "information channel of the goal term", not "reward channel". 2607.21273 ("The Dark Room in the Reward Channel") uses "reward channel" for the delivery channel.

### B. Candidate claims

| # | Claim | Closest prior and the difference | Hostile rejection | Score |
|---|---|---|---|---|
| 1 | Goal-term information channel selects evidence acquisition (matched form, same init) | SENTINEL 2606.12908, 2601.22607 and VeriGUI 2604.05477 train with state or asymmetric rewards, but never inject lying acks or ablate the channel. AgentRobustBench defers silent no-op. | "VoI arithmetic"; effective-data confound (the ack arm has more flat groups) | **4** |
| 2 | Acquisition ≠ integration ≠ reporting in a trained policy; retry depth = training depth | 2610.07753 (frontier models act without using evidence; eval only). 2609.16268 (RL invokes tools on cues). 2607.03478 (RL selects among SFT conditional policies). | "Known RL-selects-SFT result"; one tool, one domain | **4** |
| 3 | A state-grounded honesty term suppresses honest failure under GRPO (M1 0.00 vs M2 0.30) | 2608.00301 (group normalisation replaces designed penalties). GDPO 2601.05242 and CorrGRPO 2609.36820 (multi-reward domination). GracefulRL 2610.20401. None shows this in a tool agent with side effects. | k = 1; M2b pending; mechanism known | **2 now, 4 if replicated** |
| 4 | Honest failure rode along with a report-blind dense state reward | GracefulRL: fallback emerges, but it is rewarded | drift, not selection; one seed | 3 |
| 5 | RL cannot select an unsampled behaviour (0/48 → 0/150) | 2504.13837; 2607.03478 | fully anticipated; use as motivation | 1–2 |
| 6 | LLM-judge reward saturates; judges accept false completions | Confident Closing; 2607.05904 | scooped | 1 |
| 7 | Harness read-after-write text degrades the RL policy | 2608.02645, 2608.19303 (wrappers on untrained models) | one seed, one mapping | 3 |

### C. Gaps

**Sharpest gap: the training-time counterpart of the false-success literature.**
- Every false-completion, overclaim or silent-failure paper evaluates frozen models: LIMBO, Confident Closing, FTA, OverclaimBench, 2611.10063, 2609.14758.
- Every agentic-RL paper with state rewards uses visible faults: SENTINEL, 2601.22607, Fission-GRPO 2601.15625, NoisyAgent, COVERT 2604.09813, AgentRobustBench.
- Nobody asks what a policy learns when the fault is invisible to it but visible to the reward, or whether what it learns is evidence-conditioned.

**Runners-up:**
- the honesty term interacting with group normalisation, in agents with side effects;
- a metric set separating acquisition, integration and reporting on observation-equivalent controls;
- scripted retry depth as a negative transfer result.

### D. Scoop risks (quotes ≤ 15 words)

| Paper | Quote | How we differ |
|---|---|---|
| AgentRobustBench 2605.11928 | "DR-Full converts some giveups into successful retries" | They defer silent no-op |
| GracefulRL 2610.20401 | "reshapes the optimization landscape to promote graceful failure" | Not tools; the fallback is rewarded |
| 2608.00301 | "group normalization silently replaces every designed penalty with an effective penalty of one" | Ours is an instance of this mechanism, not a discovery |
| 2610.07753 | "how agents use established evidence when deciding and executing actions" | Eval only |
| 2609.16268 | "invoke external tools based on superficial prompt patterns" | Ours is scripted recovery |
| 2607.03478 | "RL training then selects for policies that obtain the highest reward" | Selection framing predates ours |
| Confident Closing / LIMBO | — | Their false-success and overclaim metrics predate our HFR/FS |

### E. Suggested CPU analyses

Status of each, run in `scripts/a6_evidence_use.py`:

| # | Analysis | Status |
|---|---|---|
| 1 | Reward-term frequency audit | open: needs training rollouts |
| 2 | Evidence → report matrix | **done** |
| 3 | Retry structure | **done** |
| 4 | F4 status confusion | **done** |
| 5 | A6 zero-variance share split by term | open |

---

## Scout 2: further gaps

| # | Gap | Nearest work and the difference | Our evidence | Feasibility | Novelty |
|---|---|---|---|---|---|
| G1 | **Honesty-term paradox**: a rare-event report penalty suppresses the behaviour it rewards under group normalisation | GDPO, CorrGRPO, RDPO 2605.13641 (heterogeneous reward normalisation), Dark Room 2607.21273. ToolRLA 2603.01620 proposes multiplicative decomposition, a ready ablation. None documents a sign flip for honesty terms. | A6 seed 0 (0.30 → 0.00) | CPU: per-term advantage contribution from logged group rewards; one retrain with per-term normalisation or a multiplicative honesty gate | 4 |
| G2 | **Discovery–exploitation gap in RL-trained policies** ("read but ignored") | "Agents Explore but Agents Ignore" 2604.17609 (discovery 79–81% vs exploitation 37–50%; prompted/SFT agents only). ToolFailBench 2607.04686 names `result_ignore`. AgentAbstain 2607.10059. Nobody shows outcome RL creates or fails to remove the gap. | `credits: []` read in 68/87 v2 held-out episodes with 0 retries; A6 refund-UR M1 reads `refunds: []` 50/50 and reports completed 50/50 | CPU: discovery / exploitation / retry rates per tool, before vs after GRPO | 4 |
| G3 | **Depth-of-recovery script**: learned retry is a fixed-k macro, not a conditional policy | Shortcuts and Boundaries 2611.12865; LLMs Gaming Verifiers 2604.15149 ("isomorphic verification"). Nobody tests learned recovery depth. | A6: every retry follows a read, but the cap is 4 (3 for outcome); depth 3 (5 attempts) fails for all | eval only (have it) | 4 |
| G4 | Transactional runtimes assume the model behaves; no one trains the policy against them | Atomix 2602.14849, Cordon 2606.17573, 2608.13900, Tool Receipts 2603.10060, Notarized Agents 2606.04193, Proof-or-Stop 2607.14890, Harness value 2609.20474 (a standalone verifier captures most false-pass benefit). All runtime-only. | receipt ledger; A5 certificates; middleware result | moderate | 3 |
| G5 | **Zero-VoI proposition**: an ack-measurable reward gives the verify action zero expected advantage | VeriCUA 2610.26469 (RL judge probing state), IGPO 2510.14967 and 2606.03135 (information-gain rewards); 2604.01434 (unverified). No crisp statement for actor RL. | sampled verification 0.25 → 0.10 under ack rewards | CPU: one-paragraph proposition | 3 |
| G6 | Cross-lingual retention of safety behaviour (verification and honesty), not of task actions | 2608.11110 (COLM 2026: 71–73% action retention across 41 languages) | HI 0.775, Hinglish 0.86 (LOGGED-ONLY) | cheap eval | 3 |

**New scoop risks:**
- **2604.17609**: discovery vs exploitation in prompted agents.
- **2611.12865** (2 days old): "minority conflict exposure can still lead to widespread shortcut acquisition".
- **Anthropic August 2026 Risk Report**: lists "retrying a failed action many times" and fabricating rather than flagging discrepancies as observed RL-environment behaviours. It publishes no training recipe.

**Framing proposed by scout 2.** "The goal term's observation channel decides whether verification can be learned at all (zero-VoI proposition); once learned, honesty terms fail by base-rate asymmetry under group normalisation, not by being too weak. Acquisition ≠ integration is the empirical bridge."

**Caveat added after our own trace analysis.** At evaluation the base-rate mechanism is not visible: 0 underclaims on 234 committed timeouts for every arm. G1 needs training-time evidence before it can be stated.

---

## References

**Scout 1, verified:**
- Hidden commit state and false completion (evaluation): 2609.29095 LIMBO · 2609.29522 · 2610.41453 EGR · 2610.15884 RetryBench · 2609.35732 FTA · 2606.09863 Confident Closing · 2611.10063 · 2609.14758 · 2609.20812 OverclaimBench · 2609.08589 · 2609.20211 · 2610.07753 · 2608.25553
- Runtime wrappers and contracts: 2608.02645 · 2608.19303 · 2608.23623 ECT
- Verification learning and agent RL: 2604.05477 VeriGUI · 2606.12908 SENTINEL · 2601.22607 · 2605.27209 NoisyAgent · 2605.11928 AgentRobustBench · 2604.09813 COVERT · 2601.15625 Fission-GRPO
- Reward design and reward hacking: 2605.02964 · 2607.02599 · 2604.02869 · 2608.11772 · 2607.11022
- Honesty and abstention RL: 2509.25760 TruthRL · 2511.11500 · 2608.00301 · 2601.20126 · 2507.16806 RLCR · 2610.18614 · 2610.20401 GracefulRL · 2610.06439
- Multi-reward GRPO: 2601.05242 GDPO · 2609.36820 CorrGRPO · 2607.21273
- Selection, shortcuts and misc: 2602.15515 · 2607.03478 · 2609.16268 · 2608.26372 · 2606.25819 · 2607.01084 · 2608.28128 · 2504.13837

**Scout 1, unverified:** 2602.10090 · 2607.05904 · 2605.02801 · 2606.28733 · 2607.10059 · 2603.03205 · 2610.28811 · 2609.09448

**Scout 2, verified:**
- Papers named in the gaps above: 2603.01620 · 2605.13641 · 2604.17609 · 2607.04686 · 2611.12865 · 2606.17573 · 2608.13900 · 2603.10060 · 2606.04193 · 2607.14890 · 2604.15149 · 2608.11110 · 2606.03135 · 2602.14849 · 2609.20474 · 2607.10059 · 2510.14967 · 2610.26469
- Also opened (not cited above): 2607.17641 · 2607.25091 · 2607.07405 · 2606.26080 · 2604.02500 · 2606.26300 · 2606.28430 · 2610.03984 · 2606.04990 · 2602.03412
- Anthropic August 2026 Risk Report (PDF)

**Scout 2, unverified:** 2604.01434 · 2605.30201 · 2608.05179 · 2609.23490 · 2608.27487 · 2610.26836 · 2607.05391 · 2610.19298 · 2606.03895 · 2608.01710 · 2512.17259 · 2609.01939 · 2610.15991 · 2607.12650 · 2607.03702 · 2605.14440
