"""Publish the procurement study (ProcureSim; Stages 2-4) to the Hugging Face Hub.

  stage   build build/hf_stage/: one model card per adapter, the ProcureSim task splits, the evaluation traces and
          result files, and the dataset cards (from the repository's tracked files)
  upload  push every adapter in build/hf_stage/models.json with its card, the two datasets, and a collection
    python scripts/publish_procure.py stage
    python scripts/publish_procure.py upload [--only REPO ...] [--dry-run]
Run both on the cluster copy, where the adapters (runs/) and the Hub login are. Public repos, created with exist_ok;
re-uploads replace files.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build/hf_stage"
USER, BASE = "atharvgaur18", "CohereLabs/tiny-aya-global"
CODE = "https://github.com/atharvgaur1845/actionproof-v2/tree/stage4"
PFX = f"{USER}/actionproof-proc-"

# run dir -> (repo suffix, warm start run, title, role, registration, evaluation summary)
RUNS: dict[str, tuple] = {
    "rft_proc": ("warm-start", None, "Warm start (supervised fine-tuning)",
                 "The shared warm start of every procurement GRPO run: SFT on scripted procurement episodes, 30% from a "
                 "verifying oracle and 70% from an acknowledgement-trusting script.", "PREREGISTRATION_A12.md", None),
    "rft_proc_v0": ("id1-warm-start-noverify", None, "ID1 warm start with no verifying demonstrations",
                    "Identical to the warm start except that no demonstration reads after a write (ID1).",
                    "PREREGISTRATION_ID1.md", "results/id1/warm_v0__neutral__proc_conf.summary.json"),
}
for s in range(5):
    RUNS[f"a12_proc_rv_s{s}"] = (f"a12-state-reward-s{s}", "rft_proc", f"A12 state reward (R_V), seed {s}",
                                 "GRPO with a binary reward read from the database: 1 iff every requested step verifies.",
                                 "PREREGISTRATION_A12*.md", f"results/eval/a12/conf/a12_proc_rv_s{s}__neutral__proc_conf.summary.json")
    RUNS[f"a12_proc_ra_s{s}"] = (f"a12-ack-reward-s{s}", "rft_proc", f"A12 acknowledgement reward (R_A), seed {s}",
                                 "GRPO with a binary reward read from the tool replies: 1 iff the last acknowledgement of every "
                                 "requested write was ok.", "PREREGISTRATION_A12*.md",
                                 f"results/eval/a12/conf/a12_proc_ra_s{s}__neutral__proc_conf.summary.json")
    RUNS[f"s3_rv_mix_s{s}"] = (f"s3-recovery-start-s{s}", "rft_proc", f"Recovery-start training (RSC), seed {s}",
                               "GRPO with R_V on a pool of half full episodes and half recovery starts (a scripted history "
                               "ending in a faulted write, replayed into the environment and the conversation). A known "
                               "curriculum, not a new optimiser.", "PREREGISTRATION_S3.md",
                               f"results/s3/conf/s3_rv_mix_s{s}__s3_conf_full.summary.json")
    RUNS[f"s3_matched_s{s}"] = (f"s3-matched-control-s{s}", "rft_proc", f"Class-matched full-episode control, seed {s}",
                                "GRPO with R_V on full episodes carrying exactly the recovery starts' faults, prompts and "
                                "worlds (the start-state control).", "PREREGISTRATION_S3_ADDENDUM1.md and _ADDENDUM2.md",
                                f"results/s3/conf/s3_matched_s{s}__s3_conf_full.summary.json")
for s in range(2):
    RUNS[f"id1_rv_s{s}"] = (f"id1-state-reward-s{s}", "rft_proc_v0", f"ID1 state reward from the no-verification warm start, seed {s}",
                            "GRPO with R_V from a warm start that never reads (ID1: does the reward create or select checking?).",
                            "PREREGISTRATION_ID1.md", f"results/id1/id1_rv_s{s}__neutral__proc_conf.summary.json")
    RUNS[f"s3_rb_mix_s{s}"] = (f"s3-rebalanced-recovery-start-s{s}", "rft_proc", f"Workflow-rebalanced recovery-start training, seed {s}",
                               "RSC with partial payments spread evenly across workflows in the pool (S3 addendum 2 Part B).",
                               "PREREGISTRATION_S3_ADDENDUM2.md", f"results/s3/conf/s3_rb_mix_s{s}__s3_conf_full.summary.json")
    RUNS[f"s3_rb_matched_s{s}"] = (f"s3-rebalanced-control-s{s}", "rft_proc", f"Workflow-rebalanced class-matched control, seed {s}",
                                   "The class-matched control on the rebalanced pool (S3 addendum 2 Part B).",
                                   "PREREGISTRATION_S3_ADDENDUM2.md", f"results/s3/conf/s3_rb_matched_s{s}__s3_conf_full.summary.json")
    RUNS[f"s4_fca_s{s}"] = (f"fca-dropped-dev-s{s}", "rft_proc", f"FCA (dropped at development), seed {s}",
                            "Forked counterfactual action credit. It failed its kill criterion fixed before training and was "
                            "dropped; it was never evaluated on test data. Published for transparency only.",
                            "docs/research/algorithm/FCA_GRPO.md", f"results/s4/dev/s4_fca_s{s}__s3_rec_dev.summary.json")


def summary_line(path: str | None) -> str:
    if not path or not (ROOT / path).exists():
        return "See the claims ledger in the code repository for this model's registered results."
    d = json.loads((ROOT / path).read_text())
    pr = d.get("phantom_readback")
    extra = f"; phantom read-back {pr['num']}/{pr['den']}" if isinstance(pr, dict) and pr.get("den") else ""
    split = d.get("split") or Path(path).name.split("__")[-1].replace(".summary.json", "")
    return (f"Greedy evaluation on `{split}` (n = {d.get('n')}): verified safe completion **{d['vsc']:.3f}**{extra}. "
            f"Source: `{path}`.")


def card(run: str) -> str:
    suf, warm, title, role, prereg, summ = RUNS[run]
    chain = (f'model = PeftModel.from_pretrained(base, "{PFX + suf}")' if warm is None else
             f'base = PeftModel.from_pretrained(base, "{PFX + RUNS[warm][0]}").merge_and_unload()  # warm start\n'
             f'model = PeftModel.from_pretrained(base, "{PFX + suf}")')
    trained = "" if warm is None else (
        f"\nTrained as a fresh LoRA (r = 16) on the warm start [`{PFX + RUNS[warm][0]}`](https://huggingface.co/{PFX + RUNS[warm][0]}) "
        "merged into the base. TRL 1.7.1 GRPO through AgentTune: 200 steps, 4 prompts x 8 rollouts, lr 2e-5, KL 0.02, "
        "T = 1, 16 turns (see `run_config.json`).\n")
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
tags: [lora, grpo, tool-use, agents, reinforcement-learning, actionproof, procuresim]
---
# ActionProof (procurement study): {title}

LoRA adapter on [{BASE}](https://huggingface.co/{BASE}) from the ActionProof procurement study: GRPO-trained tool agents in
ProcureSim, where a write's acknowledgement is byte-identical whether it committed, partly committed, duplicated or did
nothing, and a state-only verifier scores the database.

**Role.** {role}
{trained}
**Registration.** `{prereg}` in the [code repository]({CODE}) (hashed before training).

**Result.** {summary_line(summ)} Every published number is traced in `docs/final/FINAL_CLAIMS.md` and `claims/claims.yaml`.

## Load

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("{BASE}", torch_dtype="bfloat16")
{chain}
```

The environment, prompts, verifier and evaluation loop are in the code repository. Research artifact; simulated
procurement only; not for real payments.
"""


DATA_CARD = f"""---
license: cc-by-4.0
tags: [agents, tool-use, benchmark, reinforcement-learning, actionproof, procuresim]
---
# ActionProof ProcureSim splits

Frozen task splits of ProcureSim, the procurement simulator of the ActionProof study. Each task holds the prompt, the
initial database, the fault schedule (phantom, partial with ratio, lost acknowledgement, timeout before commit,
duplicate delivery, never commits) and, for recovery starts, the scripted prefix. Acknowledgements are byte-identical
across outcomes by design; only the database tells them apart.

- `proc/`: A12 (`proc_*`), A14/A15 forks, Stage 3 (`s3_*`, incl. recovery starts and the class-matched pool),
  addendum 2 targeted splits (`s3_pp_*`), Stage 4 shifts (`s4_*`: Hindi, Hinglish, paraphrase, renamed API, four-step
  workflow, two faults).
- `opsim/`: the sibling IT-operations and healthcare-administration domains.

Generator, environment, verifier and the hash manifests that freeze every split (`results/*_hashes.json`) are in the
[code repository]({CODE}). Synthetic data only.
"""
TRACES_CARD = f"""---
license: cc-by-nc-4.0
tags: [agents, tool-use, evaluation, reinforcement-learning, actionproof, procuresim]
---
# ActionProof procurement study: evaluation traces and results

One JSON line per evaluation episode (tool calls, acknowledgements, final state, report and verifier score) for every
registered evaluation of the procurement study, plus the statistics files behind every number. Paths mirror `results/`
in the [code repository]({CODE}); `docs/final/FINAL_CLAIMS.md` maps each published claim to its file.

Model outputs come from Cohere Tiny Aya Global adapters (CC-BY-NC-4.0, Cohere Labs).
"""
TRACE_DIRS = ["results/eval/a12/conf", "results/s3/conf", "results/s3/pp", "results/s3/dev", "results/s4/eval",
              "results/s4/dev", "results/id1", "results/a15", "results/a16", "results/a17", "results/k1", "results/s2"]
TRACE_FILES = ["results/s3/s3_summary.json", "results/s3/s3_addendum.json", "results/s3/s3_addendum_full_pd.json",
               "results/s3/s3_addendum2.json", "results/s3/s3_seedlevel.json", "results/s3/s3_review_numbers.json",
               "results/s3/s3_by_workflow.json", "results/s4/s4_summary.json", "results/s4/s4_diagnostics.json",
               "results/s4/fca_dev_summary.json"]


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "cards").mkdir(parents=True)
    models = {}
    for run, v in RUNS.items():
        (STAGE / "cards" / f"{run}.md").write_text(card(run))
        models[run] = PFX + v[0]
    (STAGE / "models.json").write_text(json.dumps(models, indent=1))
    d = STAGE / "procuresim"
    shutil.copytree(ROOT / "data/bench/proc", d / "proc")
    shutil.copytree(ROOT / "data/bench/opsim", d / "opsim")
    (d / "README.md").write_text(DATA_CARD)
    t = STAGE / "traces"
    for x in TRACE_DIRS:
        if (ROOT / x).exists():
            shutil.copytree(ROOT / x, t / x, ignore=shutil.ignore_patterns("*.npz", "*.pt", "rollouts", "pilot"))
    for x in TRACE_FILES:
        (t / x).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / x, t / x)
    shutil.copytree(ROOT / "docs/final", t / "docs/final")
    shutil.copy(ROOT / "claims/claims.yaml", t / "claims.yaml")
    (t / "README.md").write_text(TRACES_CARD)
    size = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file())
    print(len(models), "model cards;", round(size / 2**20), "MB staged")


def upload(only: list[str], dry: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    models = json.loads((STAGE / "models.json").read_text())
    for run, rid in models.items():
        if only and rid not in only:
            continue
        src = ROOT / "runs" / run / "final"
        if not (src / "adapter_model.safetensors").exists():
            print("missing", run)
            continue
        files = [src / "adapter_model.safetensors", src / "adapter_config.json", ROOT / "runs" / run / "run_config.json",
                 ROOT / "runs" / run / "log_history.json"]
        if dry:
            print("would upload", rid, [f.name for f in files if f.exists()])
            continue
        api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
        for f in files:
            if f.exists():
                api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=rid)
        api.upload_file(path_or_fileobj=str(STAGE / "cards" / f"{run}.md"), path_in_repo="README.md", repo_id=rid,
                        commit_message="Model card")
        print("model", rid, flush=True)
    if only or dry:
        return
    for name, folder in (("splits", "procuresim"), ("traces", "traces")):
        rid = PFX + name
        api.create_repo(rid, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=str(STAGE / folder), commit_message="ActionProof data")
        print("dataset", rid, flush=True)
    title = "ActionProof: procurement study"
    hit = [c for c in api.list_collections(owner=USER) if c.title == title]
    c = hit[0] if hit else api.create_collection(
        title=title, description="Tiny Aya Global adapters, ProcureSim splits and evaluation traces from the registered "
        "ActionProof procurement study.", private=False)  # Hub limit: 150 characters
    for rid in list(models.values()):
        if api.repo_exists(rid):
            api.add_collection_item(c.slug, item_id=rid, item_type="model", exists_ok=True)
    for rid in (PFX + "splits", PFX + "traces"):
        api.add_collection_item(c.slug, item_id=rid, item_type="dataset", exists_ok=True)
    api.add_collection_item(c.slug, item_id=f"{USER}/actionproof", item_type="space", exists_ok=True)
    print("collection", c.slug)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("stage", "upload"))
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stage() if a.cmd == "stage" else upload(a.only, a.dry_run)
