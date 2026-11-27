# Final competitor intelligence: 20 public Lexsi 2026 repositories

Refreshed 2026-11-28, 02:00–02:15 IST, on branch `final-sprint/winning-submission`. This file updates
`COMPETITOR_FORENSICS_TOP20.md` (audited 2026-11-27, 05:06–05:25 IST). That file still holds the full profiles;
each card here says what is new.

## Scope and method

**The set.** The 20 repositories in `ActionProof_RANK1_ZERO_RESTRAINT_EXECUTION_PROMPT.md` §2. They come from a
public search. They are **not** a verified finalist list.

**Checked live today, for all 20:**
- The default branch, its HEAD SHA and commit date (`gh api repos/<o>/<r>/commits`).
- The last push time (`gh api repos/<o>/<r>`).
- The commits between yesterday's recorded HEAD and today's (`gh api repos/<o>/<r>/compare/<old>...<new>`).
- The most recent commit on every non-default branch, for the repositories pushed after their main HEAD (02, 03,
  07, 12, 14).

**Every known Space**, from the Hub API without credentials: SDK, runtime stage, hardware, last modification.

**Deep inspection, top 8 plus 12.** A fresh clone of 01–07, 12 and 13 (depth 40, blobs above 2 MB skipped) in the
session scratchpad. We read:
- the current README head;
- the full commit messages of the new commits;
- the bibliography diffs;
- the paper and media files that exist;
- the submission checklists where they exist.

**What was not done.**
- No competitor code was run; no result file was recomputed.
- No Space was driven in a browser; no video was watched.
- Cloned content was treated as data. No instruction found in it was followed.

**Arxiv check.** The citation IDs that competitors added yesterday were checked against arXiv's own API. The
titles match (§ "Cross-cutting finding" below).

## Cross-cutting finding: the organisers' citation and paper review (new since yesterday)

Four of the 20 changed their paper or README on Nov 27 in response to what their commits call an organisers'
request or review. Commit messages, quoted:
- 05 `02f3331`: "Organisers' review: the five requested arXiv citations (kits and Aya Expanse) at first mention and
  where each kit is described; every named method, benchmark and tool cited from a fetched source; unnumbered
  Ethics Statement after Limitations".
- 03 `89bc4f7`: "Paper: cite CuratorKIT, AuditKIT, AgentTune and Tiny Aya papers (organiser request)".
- 04 `1a6ed21`: "Cite the AuditKIT paper … It now cites the arXiv paper (2610.22511)".
- 12 `c479b1e`: "add author contributions and an LLM usage statement".
- 05 `5a651fe`: "Organiser check: Hugging Face cited at first mention; … README headline names the post-trained
  models".

The pattern: each team cites the arXiv paper of every Lexsi kit it uses, plus its Cohere model's paper. 05 added
the five that match its kits (Aya Expanse, CuratorKIT, SafeTune, CircuitKIT, AuditKIT). 03 added the four that
match its kits.

**The IDs, checked on arXiv today:**

| Paper | arXiv | Title on arXiv |
|---|---|---|
| AgentTune | 2610.13335 | AgentTune: A Unified Training Lifecycle for Tool-Using LLM Agents |
| AuditKIT | 2610.22511 | AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems |
| CuratorKIT | 2606.21631 | CuratorKIT : Data Curation and Synthetic Data Generation for LLM Post-Training |
| CircuitKIT | 2607.19317 | CircuitKIT : Circuit Discovery, Evaluation, and Application Toolkit for Mechanistic Interpretability |
| SafeTune | 2609.22153 | SafeTune: A Unified Faithful Library for Auditing and Repairing Safety Drift in Fine-Tuned LLMs |
| Tiny Aya | 2603.11510 | Tiny Aya: Bridging Scale and Multilingual Depth |
| Aya Expanse | 2412.04261 | Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier |

**ActionProof today.**
- We use AgentTune, AuditKIT and CuratorKIT, and attempted CircuitKIT (WRITEUP §10).
- P5 uses Aya Expanse 8B; the prompted baselines use Aya Expanse 8B and 32B.
- `paper/refs*.bib`, the write-up and the README cite none of these arXiv papers. A grep for each ID finds nothing.
- Tiny Aya is cited only through its model card (`cohere2026tinyaya`).

This is the largest new compliance risk found in this refresh. We did not see the organisers' message itself; it
probably went to finalists by email or Discord (13's checklist names the `#hackathon-lexsi` Discord channel).
The author should look for it.

**Second pattern: a Stage 2 short paper.**
- 02, 03, 04, 05 and 12 each ship a Stage 2 paper PDF in the ACL/EMNLP template.
- 07 has paper drafts on branches.
- ActionProof's `paper/main_v3.tex` is in the EMNLP template, but it is the companion theory paper (group
  normalisation). It is not a paper of the TXN benchmark and recovery-agent result.

## Status on 2026-11-28 (live)

| # | Repository | Default-branch HEAD (commit date, UTC) | Recorded 2026-11-27 | Changed? |
|---|---|---|---|---|
| 01 | beingamanforever/Lost-in-injection | `d06a5a7` (11-24 09:44) | `d06a5a7` | No |
| 02 | Ashish-Soni08/lexsi-escalation-safe-agent | `c2a5da5` (11-25 17:50) | `c2a5da5` | Main no; branch `claude/project-thread-xhz561` `e38fff6` (11-26 19:09) |
| 03 | CyberRik/desi-agent | `89bc4f7` (11-27 14:39) | `1ae2609` | **Yes**: 1 commit, paper citations |
| 04 | dev-willbird1936/drift-monitor | `af781ec` (11-27 19:53) | `a258d4c` | **Yes**: 2 commits, AuditKIT citation, HF collections |
| 05 | itssaharsh/relapse | `5a651fe` (11-27 18:01) | `f4ed78d` | **Yes**: 5 commits, organisers' review, 54 files |
| 06 | PRINCE2-AI/TinyAya-Hinglish-Agent | `36ef775` (11-18 06:44) | `36ef775` | No (no Stage 2 commits) |
| 07 | lost-in-refusal/lexsi-multilingual-safety | `2b240ca` (11-25 13:51) | `2b240ca` | Main no; branch `claude/audit-third-model-9edb5a` `a238926` (11-27 04:40) |
| 08 | khamirdesai/evidencelock-track3 | `6c666c1` (11-16 07:52) | `6c666c1` | No |
| 09 | SushovitNanda/Safe-from-Fine-Tune-to-Phone | `b6fda2d` (11-15 12:14) | `b6fda2d` | No |
| 10 | dhairya-pandya/Lexsi-AI-Alignment | `d0f2c62` (11-20 09:55) | `d0f2c62` | No |
| 11 | Bhaveshmeghwal21/against-the-safety | `c5a093c` (11-16 10:01) | `c5a093c` | No |
| 12 | Hydra-Of-Malice/same-model-different-language | `c479b1e` (11-27 19:46) | `ea1b7fa` | **Yes**: 2 commits, paper (tracks, cost per 1,000 replies, contributions, LLM-use statement) |
| 13 | shro-2002/rewardshield | `35a85df` (11-22 07:09) | `35a85df` | No |
| 14 | AndrewPrakash1/veriloop | `7f17615` (11-21 08:04) | `7f17615` | Main no; branch `phase4-eval-cpu` `8d6a9b7` (11-21 17:53) |
| 15 | Swateya03/reward-hacking-observatory | `b1a8cb9` (11-26 20:22) | `b1a8cb9` | No |
| 16 | wenbinio/lexsi-contextual-pseudonymizer | `61b5321` (11-15 12:52) | `61b5321` | No |
| 17 | SS-360/lexsi-hackathon-tamil-clinical-triage | `f56334a` (11-12 17:01) | `f56334a` | No |
| 18 | Incharajayaram/icsa-grounded-assistant | `44032d3` (11-15 08:21, branch `master`) | `44032d3` | No |
| 19 | dixitkeshav/finsafe | `8ddaf37` (11-15 06:01) | `8ddaf37` | No |
| 20 | qasimjethwa/recovery-agent-taught-by-rules | `4df7da0` (11-13 06:37) | `4df7da0` | No |

**Spaces (Hub API, 2026-11-28 ~02:05 IST).**

| Space | SDK | Stage | Hardware | Last modified | Type |
|---|---|---|---|---|---|
| ashish-soni08/escalation-safe-agent-dashboard (02) | static | RUNNING | — | 11-15 | replay |
| ToolW/desi-agent-lab (03) | static | RUNNING | — | 11-24 | replay |
| dev-willbird1936/drift-monitor (04) | static | RUNNING | — | 11-16 | replay plus a live tab |
| dev-willbird1936/drift-monitor-scorer (04) | gradio | RUNNING | zero-a10g | 11-16 | **live inference** |
| itssaharsh/relapse (05) | static | RUNNING | — | 11-27 | results explorer |
| itssaharsh/relapse-article (05) | static | RUNNING | — | 11-27 | article |
| primce/hinglish-expense-agent (06) | gradio | **SLEEPING** | — | 11-12 | live inference, asleep |
| primce/hinglish-episodes, hinglish-attacks, hinglish-track3-article (06) | static | RUNNING | — | 11-12 to 11-15 | replay, article |
| lost-in-refusal/lost-in-refusal-demo (07) | static | RUNNING | — | 11-06 | static, predates Stage 2 |
| Earther/evidencelock-demo (08) | static | RUNNING | — | 11-15 | replay |
| BMEscanor/against-the-safety-trace-viewer (11) | static | RUNNING | — | 11-05 | replay |
| Finsafe-team/finsafe (19) | static | RUNNING | — | 11-15 | static |
| qasimjethwa/recovery-agent-taught-by-rules (20) | static | RUNNING | — | 11-08 | replay |
| *atharvgaur18/actionproof-live (ours)* | gradio | RUNNING | zero-a10g | 11-27 | **live inference** |
| *atharvgaur18/actionproof (ours)* | static | RUNNING | — | 11-27 | replay study |

No public Space was found for 01, 12, 13 or 14 under their account names. 12 keeps Space source in `space/`
(including `requirements-live.txt`); 14 keeps a live-demo commit on an unmerged branch.

Only two of these Spaces run a model live, and both are awake: ours (Track 3, a Cohere agent) and 04's scorer
(Track 1, a classifier behind an agent). 06's live Space is asleep.

---

## The 20 cards

Fields follow prompt §2. "Unchanged" means the full profile in `COMPETITOR_FORENSICS_TOP20.md` still stands, and
it is condensed here.

### 01 Lost in Injection
- **HEAD.** `d06a5a7`, 2026-11-24. Unchanged since yesterday; no new branch work.
- **Problem and contribution.** Indirect prompt injection across 11 languages, scripts and code-mixes. Shows that
  Meta-SecAlign-style defenses transfer one language at a time and leak through filters, rewording attackers and a
  first-word scorer.
- **Model, toolkits, track.** Tiny Aya Global, Fire, Earth and Water, plus Qwen3-4B, Llama 3.1 8B / 3.3 70B and
  Meta-SecAlign 8B / 70B. CuratorKIT, AlignTune and AuditKIT are imported. Track 2a.
- **Strongest claim.**
  - Against the English-only Tiny Aya defense, trying all 11 languages hijacks 21.3% of cases (95% CI 15.9–26.9;
    three seeds 20.2 / 23.1 / 20.7), against 97.1% undefended.
  - Low-resource training cuts this by −7.1 points [−13.6, −0.8].
  - Scale: 208 AlpacaFarm-Hacked cases × 11 languages, 7 models, 940,992 scored replies.
- **Baselines.** Equal-budget paraphrase and noise controls; three scoring rules. Fair.
- **Novelty vs engineering.** A cross-lingual transfer finding plus a scorer-sensitivity result. The defense recipe
  is borrowed (stated).
- **Demo device.** An `attack.gif` hero, a 4-panel findings figure, an 87 s Manim video and a slides PDF.
- **Deployment.** None; a judge cannot run anything.
- **Reproducibility.** Per-stage bash scripts and pinned uv environments. It needs a 24 GB GPU, gated Llama and
  Meta-SecAlign models, and an OpenRouter judge key.
- **Strongest limitation.** The headline judge is one LLM, not validated by humans. Agent benchmarks sit near
  their floor (under 4% even undefended).
- **Lesson for us (<24 h).** Lead with three or four numbered findings, each tied to one figure. Add one
  scoring-rule sensitivity table: the same episodes under different success definitions.
- **Do not copy.** Multilingual breadth (not our problem) and 70B-scale rows.

### 02 Escalation-Safe Tool-Using Aya
- **HEAD.** Main `c2a5da5`, 2026-11-25; unchanged. The newest branch commit is `e38fff6` (2026-11-26 19:09,
  `claude/project-thread-xhz561`): "paper: drop pending hand-label mentions; agreement is judge-to-judge only". So
  02 dropped its planned human labels; the detector is checked only against LLM judges (κ 0.87 / 0.84).
- **Problem and contribution.** Tool agents meet bot walls, logins, captchas and rate limits. They should take a
  legitimate route, or stop and say so. The contribution is a reward in which every exploit-shaped call scores below
  every honest outcome.
- **Model, toolkits, track.** Tiny Aya Global, Fire, Water and Earth, and Aya Expanse 8B. AgentTune, AuditKIT,
  CircuitKIT and CuratorKIT are imported. Track 3. The README opens with a "Lexsi kits used" line.
- **Strongest claim.**
  - Same start model, same 160 GRPO steps: escalation on blocked test tasks 0.0% (safe reward v4.1) against 99.2%
    (success-only reward).
  - Sealed set: 16.4 ± 4.7 (three seeds) against 100.0.
  - 320 tasks per set; seven sets, including a real HTTP server.
- **Baselines.** Reward ablations at one seed each; clean SFT; success-only GRPO. Fair, with the same start model.
- **Novelty vs engineering.** Reward design against shortcut bypass. Their "escalation" is observable: the block is
  visible in the tool output.
- **Demo device.** A per-set results table on the README front page. The static dashboard has an episode browser.
- **Deployment.** Static dashboard, RUNNING. Replay only.
- **Reproducibility.** Best in class for number checks: a ~5-minute no-GPU route regenerates the data files
  byte-identically and expects an empty `git status`. Each training command is an HF Jobs line with its cost.
- **Strongest limitation.**
  - Unseen shortcut kinds are only partly solved (16 ± 5% sealed, 23 ± 5% unseen).
  - The detector was gamed twice.
  - It over-refuses on 43% of fresh-set tasks.
  - No human labels.
- **Lesson for us (<24 h).** A one-line finding followed at once by one table across all held-out sets. A
  "kits used" line at the top of the README.
- **Do not copy.** The framing of escalation as refusal of a visible block; our case is hidden execution state.
- **Threat.** Highest in Track 3. It owns "stop and say so"; our SEL1 hand-off result is partial.

### 03 Desi Agent
- **HEAD.** `89bc4f7`, 2026-11-27 14:39, one commit since `1ae2609`. "Paper: cite CuratorKIT, AuditKIT, AgentTune
  and Tiny Aya papers (organiser request); fix broken appendix ref": 3 files, +28 bib lines. No new result.
- **Problem and contribution.** A multilingual banking support agent (English, Hindi, Bengali, Hinglish, Banglish)
  that picks the right tool, fills exact arguments, and asks instead of guessing.
- **Model, toolkits, track.** Tiny Aya Global, with a Qwen3-14B teacher for customer text only. AgentTune, AuditKIT
  and CuratorKIT are imported. Track 3.
- **Strongest claim.** On 486 held-out conversations, every call executed against a deterministic mock bank:
  - task success 45.9% → 97.7% (SFT), 98.6% (GRPO 2);
  - wrong bank actions 20.8% → 1.2–1.8%;
  - language switch 34.0% → 78.3%.
  Two GRPO seeds; McNemar tests.
- **Baselines.** Base model, then staged SFT / RFT / GRPO. The reward was audited against scripted cheating
  policies.
- **Novelty vs engineering.** Mostly strong engineering: code-generated labels and staged distillation.
- **Demo device.** A banner carrying the two headline numbers. The Lab has a tour, a case still wrong, and an
  explorer over all 486 conversations filtered by "fixed / still failing / regressions". A tour video.
- **Deployment.** Static Lab, RUNNING; replay only, stated plainly.
- **Reproducibility.** CI, traces on the Hub; `pyproject.toml` not pinned.
- **Strongest limitation.** The test is template-generated from the training distribution; 97.7% is near the
  ceiling, and the grounding gain is partly in-distribution (stated).
- **Lesson for us (<24 h).** An outcome-filtered episode explorer ("fixed / still failing / regressions") over our
  registered test episodes. A banner with two numbers.
- **Do not copy.** Near-ceiling in-distribution headlines.

### 04 Drift Monitor
- **HEAD.** `af781ec`, 2026-11-27 19:53, two commits since `a258d4c`:
  - `1a6ed21` cites the AuditKIT arXiv paper;
  - `af781ec` links two HF collections (model and data) and states where the scripts read data from.
  No new result.
- **Problem and contribution.** A fine-tuned Tiny Aya emits one digit for "drift from what the user authorised"; a
  threshold decides when to call a frontier advisor.
- **Model, toolkits, track.** Tiny Aya Global; CuratorKIT, AlignTune and AuditKIT. Track 1, problem statement 2
  ("replacing a closed-model API").
- **Strongest claim.**
  - 770 held-out real sessions: AUROC 0.552 → 0.960; false alarms on clean steps 100% → 5.4%.
  - Closed loop of 256 sessions: unrequested files 2.48 → 0 per session; task checks 84% → 99%; 1.4 advisor calls
    per drifting session against 4.8.
  - One training seed.
- **Baselines.** Two frontier models with the same prompt on the same 497 rows; review-every-turn as the cost
  baseline.
- **Novelty vs engineering.** Product engineering with good evaluation. Counterfactual twin sessions are the same
  idea as our quadruples, applied to inputs, not hidden state.
- **Demo device.** Recorded runs replayed step by step. A "Try it live" tab runs a real agent with no key, through
  the ZeroGPU scorer and a spend-capped shared key. Idle, offline and wake states.
- **Deployment.** Static page RUNNING; scorer Gradio on ZeroGPU RUNNING (live).
- **Reproducibility.** Seeds fixed; data and model public; collections linked. No one-command number check found.
- **Strongest limitation.** One seed; model-generated labels; written, not executed, sessions; drift inside a
  requested file goes unseen.
- **Lesson for us (<24 h).** An "At a glance" block with a "Where it falls short" bullet, and a cost per million
  operations ($463 against $1,735). A finished 2:48 video with slides and captions.
- **Do not copy.** A frontier model as the advisor in the loop (not a Cohere core role).
- **Threat.** Highest on presentation, product and demo. It is not Track 3.

### 05 Relapse
- **HEAD.** `5a651fe`, 2026-11-27 18:01, five commits and 54 files since `f4ed78d`:
  - the organisers' review (five arXiv citations; every named method cited from a fetched source; an unnumbered
    Ethics Statement; Conclusion before Limitations);
  - "Audit fixes: run total 686";
  - "Title names the post-trained models";
  - "Organiser check: … README headline names the post-trained models".
  The day before, 05 also added a paper "For the kits" box and drafted one issue report per kit
  (`a5a1da9`, not posted).
- **Problem and contribution.** Does unlearning of hazardous biology survive deployment: 4-bit files, and the
  tokenizer setting a user's copy loads?
- **Model, toolkits, track.** Tiny Aya Base, Global, Fire, Water and Earth. SafeTune; CircuitKIT and AuditKIT
  imported. Track 2b.
- **Strongest claim.**
  - At SafeTune's default layers, 16 of 16 counting runs give back about half (47%) of the audited forgetting
    through the official tokenizer; at layers 3–5, 0 of 16.
  - Four-bit forgetting holds in 107 of 109 runs.
  - 686 runs; six preregistrations, 25 hypotheses (11 held, 2 in part, 11 failed, 1 untestable).
- **Baselines.** The same weights scored through both settings in the same job: a strong within-run control. The
  layer effect is confounded (stated).
- **Novelty vs engineering.** A measurement finding about a deployment artefact, backed by real-world exposure:
  418 derived Hub repositories (64,619 downloads) still carried the old setting on 2026-11-26. They ship a
  checker.
- **Demo device.** An "In brief" table; a public correction displayed prominently; `docs/demo.mp4` and
  `docs/demo.gif`; an article Space.
- **Deployment.** Static explorer and article, both RUNNING (updated 11-27).
- **Reproducibility.** HF Jobs scripts that read only public repositories; a citation fetch record
  (`paper/citations_checked.md`).
- **Strongest limitation.** A tooling artefact rather than a method; the layer effect is confounded with damage.
- **Lesson for us (<24 h).**
  - Cite every kit's arXiv paper.
  - Add a "For the kits" box: what we found about AgentTune and AuditKIT that their maintainers can use. AuditKIT's
    judge accepted 71% (213 / 298) of false completions; CircuitKIT failed to run.
  - Name the released model in the README headline.
  - Count our hypotheses as held / in part / failed.
- **Do not copy.** A 20-word title.
- **Threat.** The strongest integrity and impact story; no system to show.

### 06 Hinglish Expense Agent
- **HEAD.** `36ef775`, 2026-11-18. Unchanged; no Stage 2 commits.
- **Problem and contribution.** A Hinglish shared-expense agent (ledger, currency, settling up), with a
  deterministic CPU guard under the model.
- **Model, toolkits, track.** Tiny Aya Global; AgentTune (docs; no import found); AuditKIT via a wrapper. Track 3.
- **Strongest claim.**
  - hard-257: 0 / 257 → 235 / 257 (Wilson 87.4–94.3%).
  - hard-90: 3 / 90 → 84 / 90 (three seeds pooled 93.0%).
  - The guard keeps attacker money out in 208 / 208 injected episodes.
- **Baselines.** Base model; guard on and off; GRPO reported as a null (0 / 0).
- **Novelty vs engineering.** Engineering. The guard is a filter, not a proof (stated; blind set 5 / 8).
- **Demo device.** A numbered "Judge's 60-second view"; a CPU "try the guard" path off the GPU quota; galleries; a
  1:44 narrated video.
- **Deployment.** Live Gradio Space SLEEPING; static galleries RUNNING.
- **Reproducibility.** `verify_all.py --with-tests` runs 31 checks with no GPU and no token. A Hub hash manifest
  with `--check`.
- **Strongest limitation.** Stage 1 only; small safety splits; one base model.
- **Lesson for us (<24 h).** A "why these numbers cannot be gamed" table, mapping shortcuts to the mechanism that
  blocks each (our README has one; keep it visible).
- **Do not copy.** A guard presented without its blind-spot rate. (06 itself does report the rate.)

### 07 Lost in Refusal
- **HEAD.** Main `2b240ca`, 2026-11-25; unchanged. Work continues on branches:
  - `a238926` (11-27 04:40), registration section 23, an internal provenance note about figures;
  - `mingyu/stage2` `a182e6f` (11-26), section 22, a correction: NF4 is 3.3 points below bfloat16 on
    Global-MMLU-Lite.
- **Problem and contribution.** Benign multilingual LoRA erodes refusals (Korean, and untrained Swahili). SafeTune
  SafeGrad as the repair.
- **Model, toolkits, track.** Tiny Aya Global and Earth; Granite 4.1 3B as an outside family. SafeTune, AlignTune,
  AuditKIT and CuratorKIT imported. Track 2.
- **Strongest claim.** Korean drift in 3 of 3 seeds (Korean minus English +11.8 [+6.8, +16.7]); SafeGrad repair
  pooled −8.4 [−11.3, −5.5].
- **Baselines.** Plain against SafeGrad with identical data, order and config; a simple-fix baseline; a second
  machine and a second training stack.
- **Novelty vs engineering.** A careful replication-style study; SafeGrad is borrowed.
- **Demo device.** A table of checks and results.
- **Deployment.** Static Space, last modified Nov 6 (predates Stage 2).
- **Reproducibility.** An outputs dataset; pinned per-lane requirement files.
- **Strongest limitation.** One automatic safety judge (a second failed agreement); the human reading of the Korean
  rise did not pass (p = 0.22); SafeGrad does not beat the simple fix.
- **Lesson for us (<24 h).** Put the registration commit and time next to each result it governs. Most of our
  tables already do; check the deck.
- **Do not copy.** Prose density.

### 08 EVIDENCELOCK
- **HEAD.** `6c666c1`, 2026-11-16. Unchanged; no Stage 2 activity.
- **Problem.** An invoice agent must verify against a trusted ledger before approving, while vendor documents inject
  text. Track 3.
- **Model and toolkits.** Tiny Aya Global; AgentTune and AuditKIT.
- **Strongest claim.** Secure completion 33 → 91 / 120 [40.0, 55.8 pp]; verification before acting 60 / 96 →
  96 / 96. GRPO-2 null (216 → 216 / 264).
- **Baselines.** A CVT ablation (−52.5 pp) and pre-registered follow-ups; small panels.
- **Novelty.** Evidence-before-action is the closest idea to our read-back; theirs is observable.
- **Demo.** Static replay Space, RUNNING.
- **Reproducibility.** `pytest` runs 53 checks of the headline numbers in about one second.
- **Limitation.** Inactive in Stage 2; small n.
- **Lesson for us.** A "status" column on every headline table (we have one).
- **Do not copy.** Nothing material.

### 09 Safe from Fine-Tune to Phone
- **HEAD.** `b6fda2d`, 2026-11-15. Unchanged.
- **Problem.** Medical LoRA erodes safety; a steering repair that survives GGUF. Track 2.
- **Model and toolkits.** Tiny Aya Global; SafeTune and CircuitKIT (README).
- **Strongest claim.** Drift +18.1 [15.0, 21.3]; steering −10.1 [−12.5, −7.6]; one training run per model.
- **Baselines.** Harmful / benign twins.
- **Demo.** None.
- **Reproducibility.** A hash manifest.
- **Limitation.** One seed; no demo.
- **Lesson.** State what each bootstrap resamples (tasks, not seeds).
- **Do not copy.** Nothing.

### 10 Lexsi AI Alignment
- **HEAD.** `d0f2c62`, 2026-11-20. Unchanged.
- **Problem.** Bengali fine-tuning attacks bypass English-tested SafeTune defenses. Track 2.
- **Model and toolkits.** Tiny Aya Fire; SafeTune.
- **Strongest claim.** Bengali harm 50% → 14% by projecting out the top singular direction (k = 1), MMLU kept.
  Seeds and intervals are not stated in the README.
- **Demo.** A Vercel site and a Space `app.py` (not checked live).
- **Limitation.** README-level evidence is thin.
- **Lesson.** A hero image built on one contrast; ours is the same "ok" over two different databases.
- **Do not copy.** Unstated seeds.

### 11 Against the Safety
- **HEAD.** `c5a093c`, 2026-11-16. Unchanged.
- **Problem.** RAG answers with citations, or abstains when evidence is missing. Track 3, problem statement 4.
- **Model and toolkits.** Tiny Aya Global (pinned); AgentTune and AuditKIT 1.1.3.
- **Strongest claim.** Answers with hidden evidence 281 / 375 → 42 / 375; macro success 58.0% against **42.0% for
  refusing everything**. One run.
- **Demo.** Static trace viewer ("the model does not run").
- **Reproducibility.** Pinned CPU and GPU lock files.
- **Limitation.** One run; later training changes failed their own release rules.
- **Lesson.** An always-escalate baseline beside every escalation metric (adopted in SEL1).
- **Do not copy.** Nothing.

### 12 Same Model, Different Language
- **HEAD.** `c479b1e`, 2026-11-27 19:46, two commits since `ea1b7fa`. Paper only:
  - tracks as bullets, with Track 2a items;
  - cost per 1,000 replies;
  - author contributions and an LLM-usage statement.
  No new result.
- **Problem.** Per-language safety gaps between Tiny Aya Earth and Global; repairs without retraining. Track 2b.
- **Model and toolkits.** Tiny Aya Earth and Global; Command A as a judge; CircuitKIT and others.
- **Strongest claim.** Javanese gap +9.7 (+4.0 to +15.3), McNemar p = 0.0023, under a corrected bar of 0.0036;
  Command A confirms +10.8 (p = 0.0013).
- **Baselines.** Circuit-guided edits against edits at random layers. This is a good negative control: the circuit
  edits did no better.
- **Demo.** `space/` source with a live requirements file; no public Space found.
- **Reproducibility.** A reproduction notebook; CI; pinned-revision tables in the paper appendix.
- **Limitation.** Mostly null headline; no repair met all pre-set conditions.
- **Lesson.** The paper's cost per 1,000 replies, and an LLM-use statement in the paper itself (we have §33 in the
  write-up).
- **Do not copy.** Nothing material.

### 13 RewardShield
- **HEAD.** `35a85df`, 2026-11-22. Unchanged.
- **Problem.** False accepts by SQL-execution rewards. Track 3, problem statements 9 and 6.
- **Model and toolkits.** Tiny Aya Global; AgentTune (the verifier is a reward callable in AgentTune's rollout);
  AuditKIT.
- **Strongest claim.** False accepts 3.88% → 1.39% on dev-A. An external audit showed this depended on one database
  draw: over 200 draws the gap is −0.25 [−0.59, −0.00].
- **Baselines.** An oracle independent in code; five verifiers.
- **Demo.** None.
- **Reproducibility.** Re-derives every headline number; external audit reproduced line by line.
- **Limitation.** Tiny effects; no demo.
- **Lesson.** Our reference verifier is a consistency check (same author and spec). The README says so; keep it.
- **Do not copy.** Nothing.

### 14 VeriLoop
- **HEAD.** Main `7f17615`, 2026-11-21. Unchanged. Branch `phase4-eval-cpu` `8d6a9b7` (11-21 17:53): "a LIVE demo —
  the real model, on the GPU, streamed into the browser". Not merged; no public Space found.
- **Problem.** Compliance steps skipped by small agents; rewards from database state plus an audit log. Track 3.
- **Model.** No Cohere model or Hub artefact found on main.
- **Strongest claim.** None. Main says "a plan, not a report".
- **Lesson.** Publish per-task postconditions as machine-readable contracts.
- **Do not copy.** Nothing.

### 15 Reward Hacking Observatory
- **HEAD.** `b1a8cb9`, 2026-11-26. Unchanged.
- **Problem.** A reward-hacking taxonomy for tool agents.
- **Model.** `claude-haiku-5-5` agent; **no Cohere model**; likely non-compliant.
- **Strongest claim.** None published (fixture numbers were withdrawn).
- **Lesson.** Nothing material.

### 16 Contextual Pseudonymizer
- **HEAD.** `61b5321`, 2026-11-15. Unchanged.
- **Problem.** Identifier extraction. Track 1.
- **Model and toolkits.** Tiny Aya Global (pinned revision); CuratorKIT, AlignTune, AuditKIT.
- **Strongest claim.** Exact entity recall 66.0% → 94.6% on 120 fresh records; precision 84.6%; one run.
- **Demo.** Local only.
- **Lesson.** Pin the base model by revision in every card.

### 17 Tamil Clinical Triage
- **HEAD.** `f56334a`, 2026-11-12 (one commit). Unchanged.
- **Problem.** Clinical triage in Tamil and Tanglish. Tracks 1 and 2a.
- **Strongest claim.** Safety pass 52.5% → 97.5% on 40 probes, including a rule engine.
- **Demo.** None.
- **Lesson.** Nothing material.

### 18 ICSA Grounded Assistant
- **HEAD.** `44032d3`, 2026-11-15 (branch `master`). Unchanged.
- **Problem.** An ICS / OT advisory assistant that cites or abstains. Track 1.
- **Model.** Core model `Qwen/Qwen2.5-7B-Instruct`, not a Cohere model (Hub `base_model`); likely ineligible.
- **Strongest claim.** 12 held-out cases.
- **Lesson.** Nothing.

### 19 FinSafe
- **HEAD.** `8ddaf37`, 2026-11-15. Unchanged.
- **Problem.** Finance adaptation and refusal erosion; steering repair. Tracks 2b and 1.
- **Model and toolkits.** Tiny Aya Global (pinned); CuratorKIT, SafeTune, AuditKIT, AlignTune.
- **Strongest claim.** Valid JSON 64% → 94%; safety erosion not observed on a pilot (12 prompts per cell).
- **Demo.** Static Space, RUNNING.
- **Lesson.** A label-sensitivity analysis.

### 20 Recovery Agent, Taught by Rules
- **HEAD.** `4df7da0`, 2026-11-13. Unchanged.
- **Problem.** Failed-payment recovery plans executed as tool calls, taught by a production rule engine. Track 3;
  the closest domain to ours.
- **Model and toolkits.** Aya Expanse 8B; AgentTune (SFT).
- **Strongest claim.** Exact end-to-end success 0 / 128 → 47 / 128 (36.7%); one run; an independent oracle and a
  corrupt-prediction negative control.
- **Limitation.** The failure reason is given in the input: no hidden state, no ambiguous outcome.
- **Demo.** Static replay, RUNNING.
- **Lesson.** Name the real-world rule source: for us, the idempotency and reconciliation practice of payment APIs.
- **Do not copy.** Nothing.

---

## Top threats, ranked for overall rank 1 (subjective)

1. **02 Escalation-Safe Aya.** Track 3; owns safe stopping; best number-check route. No change since Nov 25.
2. **04 Drift Monitor.** Best product, demo and video; cost per million steps. Organiser citation fix done Nov 27.
3. **01 Lost in Injection.** Breadth and the best visual story; no demo. Inactive since Nov 24.
4. **05 Relapse.** Integrity and real-world exposure. The most active in the last 24 h (organisers' review fixed).
5. **03 Desi Agent.** Polish and a judge explorer. Organiser citation fix done Nov 27.
6. 07 Lost in Refusal, 12 Same Model Different Language, 13 RewardShield, 06 Hinglish Expense Agent.

The ranking matches yesterday's. What changed: three of the top five (03, 04, 05) cited the kit papers on Nov 27,
and so did 12. A grep for the arXiv IDs also finds them in 06 and 07, but not in 01, 02 or 13 (main branches
only). ActionProof cites none of them.
