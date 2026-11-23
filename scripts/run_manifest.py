"""Unified run manifest (docs/quality/EXPERIMENT_PIPELINE_AUDIT.md §B): one entry for **every** directory under
runs/, whether complete, incomplete, crashed or running. Run where the adapters live (the cluster).

    python scripts/run_manifest.py --out results/run_manifest.json

**Status rules.** A checkpoint never implies completion.

| Status | Rule |
|---|---|
| complete | the final adapter exists **and** the trainer's logged last step equals the configured steps (SFT: the trainer's last logged epoch reaches the configured epochs) |
| incomplete | checkpoints or a partial log, no final adapter, or a logged step below the configured steps |
| crashed | the directory is marked `.crashed*`, or its producing task is in a `failed/` queue |
| running | its producing task is in a `running/` queue |
| complete_unlogged | a final adapter exists and its task did not fail, but no final training log proves the step or epoch count (e.g. SFT runs that kept only the epoch-1 checkpoint's state, or HF-job runs copied without their config) |
| merged_copy | a merged model directory (`*_merged`), not a run |
| unknown | none of the above can be established |

**Fields** (null with a `not_recorded` note when the run did not log them): run id, experiment family and registration,
the producing task and its queue outcome (done = exit 0, failed = non-zero), base model, initial adapter and its
hash, final adapter and its hash, pool or dataset hashes, seed, reward, learning rate, β, group size, prompts per
step, rollouts per step, max turns, completion length, LoRA r / alpha / target modules / dropout, configured and
logged steps, train runtime, checkpoint paths, evaluation result files.

Not recorded by any run before P2: GPU type, peak memory, tokens generated and trained, and the git commit at training
time. They are listed under `not_recorded` rather than guessed. The manifest records the commit at build time.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# prefix -> (experiment family, registration or development document)
FAMILIES = [
    ("a12_proc_", "A12", "PREREGISTRATION_A12*.md"), ("k1_", "K1", "PREREGISTRATION_K1.md"),
    ("s3_rv_mix25", "S3 dev stage B", "docs/stage3/RESEARCH_REPORT.md"), ("s3_rv_mix", "S3 (RSC)", "PREREGISTRATION_S3.md"),
    ("s3_matched", "S3 addendum 1 (class-matched control)", "PREREGISTRATION_S3_ADDENDUM1.md"),
    ("s3_ackonly", "S3 addendum (acknowledgement-only starts)", "PREREGISTRATION_S3_ADDENDUM*.md"),
    ("s3_rb_", "S3 addendum (read-back variant)", "PREREGISTRATION_S3_ADDENDUM*.md"),
    ("s3_sr_mix", "SR1 (tool-name randomisation)", "PREREGISTRATION_SR1.md"),
    ("s3_p_", "S3 development pilot", "docs/stage3/RESEARCH_REPORT.md"),
    ("s4_fca", "FCA (development)", "docs/research/algorithm/FCA_GRPO.md"), ("s4_smoke", "S4 smoke", None),
    ("s5_c5", "C5 (development)", "docs/research/algorithm/C5G_DEV.md"),
    ("s5_guide", "G (development)", "docs/research/algorithm/C5G_DEV.md"),
    ("ad_", "AD (algorithm discovery)", "PREREGISTRATION_AD*.md"), ("id1_", "ID1", "PREREGISTRATION_ID1.md"),
    ("p1_", "P1", "PREREGISTRATION_P1.md"), ("p2_", "P2", "PREREGISTRATION_P2.md"),
    ("rft_proc", "A12 warm start (SFT)", "PREREGISTRATION_A12*.md"), ("rft_", "Stage 1 SFT", "PREREGISTRATION.md"),
    ("grpo_tiny_", "Stage 1 GRPO", "PREREGISTRATION.md (amendments)"), ("dpo_", "Stage 1 DPO pilot", "PREREGISTRATION.md"),
    ("g0_", "Stage 1 gate G0", "PREREGISTRATION.md"),
]
NOT_RECORDED = ["gpu_type", "peak_memory", "tokens_generated", "tokens_trained", "git_commit_at_training"]


def sha(p: Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def family(name: str):
    for pre, fam, reg in FAMILIES:
        if name.startswith(pre):
            return fam, reg
    return None, None


def producing_tasks() -> dict:
    """run name -> (task file, queue outcome) from `--out runs/<name>` in slurm/tasks and slurm/*.sh."""
    outcome = {}
    for state in ("done", "failed", "running", "pending", "hold"):
        for f in glob.glob(str(ROOT / f"queue/*/{state}/*.sh")):
            n = Path(f).stem
            # done beats failed (a retried task), running beats pending
            rank = {"done": 4, "running": 3, "failed": 2, "pending": 1, "hold": 0}[state]
            if rank >= outcome.get(n, ("", -1))[1]:
                outcome[n] = (state, rank)
    out = {}
    for f in sorted(glob.glob(str(ROOT / "slurm/tasks/*.sh"))):
        txt = Path(f).read_text()
        for run in re.findall(r"--out runs/([A-Za-z0-9_.\-]+)", txt):
            out.setdefault(run, (Path(f).stem, outcome.get(Path(f).stem, (None,))[0]))
    return out


def adapter_dir(run: Path) -> Path | None:
    for d in (run / "final", run):
        if (d / "adapter_model.safetensors").exists():
            return d
    return None


def last_logged(run: Path) -> tuple[int | None, float | None, float | None, str | None]:
    hist = run / "log_history.json"
    rows = json.loads(hist.read_text()) if hist.exists() else []
    src = "log_history.json" if rows else None
    if not rows:  # fall back to the newest checkpoint's trainer_state
        cks = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1)
        if cks and (cks[-1] / "trainer_state.json").exists():
            rows = json.loads((cks[-1] / "trainer_state.json").read_text()).get("log_history", [])
            src = f"{cks[-1].name}/trainer_state.json"
    step = max((x.get("step", 0) for x in rows if isinstance(x, dict)), default=None)
    epoch = max((x.get("epoch", 0) for x in rows if isinstance(x, dict)), default=None)
    rt = next((x.get("train_runtime") for x in reversed(rows) if isinstance(x, dict) and "train_runtime" in x), None)
    return step, epoch, rt, src


def entry(run: Path, tasks: dict) -> dict:
    name = run.name
    fam, reg = family(name)
    cfg_p = run / "run_config.json"
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    ad = adapter_dir(run)
    acfg = json.loads((ad / "adapter_config.json").read_text()) if ad and (ad / "adapter_config.json").exists() else {}
    step, epoch, runtime, log_src = last_logged(run)
    task, q = tasks.get(name, (None, None))
    conf_steps = cfg.get("steps") or cfg.get("max_steps")
    conf_epochs = cfg.get("epochs")
    if name.endswith("_merged") or (run / "config.json").exists() and not ad:
        status = "merged_copy"
    elif ".crashed" in name or q == "failed":
        status = "crashed"
    elif q == "running":
        status = "running"
    elif ad and conf_steps and step is not None and step >= conf_steps:
        status = "complete"
    elif ad and conf_epochs and epoch is not None and epoch >= conf_epochs - 1e-6:
        status = "complete"
    elif ad and log_src != "log_history.json" and q in ("done", None):
        # a final adapter is written only after training returns, but no final log proves the step count
        status = "complete_unlogged"
    elif list(run.glob("checkpoint-*")) or step:
        status = "incomplete"
    else:
        status = "unknown"
    pools = cfg.get("pools") or cfg.get("task_jsonl") or cfg.get("data")
    init = cfg.get("init_adapter")
    return {
        "run_id": name, "family": fam, "registration": reg, "status": status,
        "task": task, "task_queue_outcome": q,
        "base_model": cfg.get("model"), "init_adapter": init,
        "init_adapter_sha256": sha(ROOT / init / "adapter_model.safetensors") if init else None,
        "final_adapter": str(ad.relative_to(ROOT)) if ad else None,
        "final_adapter_sha256": sha(ad / "adapter_model.safetensors") if ad else None,
        "data_sha256": {p: sha(ROOT / p) for p in pools.split(",")} if isinstance(pools, str) else None,
        "seed": cfg.get("seed"), "reward": cfg.get("reward", cfg.get("arm")), "lr": cfg.get("lr"), "beta": cfg.get("beta"),
        "group_size": cfg.get("num_generations"), "prompts_per_step": cfg.get("prompts_per_step"),
        "rollouts_per_step": (cfg["num_generations"] * cfg["prompts_per_step"]) if cfg.get("num_generations") and cfg.get("prompts_per_step") else None,
        "max_turns": cfg.get("max_turns"), "max_completion_length": cfg.get("max_completion_length"),
        "epochs": conf_epochs, "configured_steps": conf_steps, "logged_last_step": step, "logged_last_epoch": epoch,
        "train_runtime_s": runtime, "log_source": log_src,
        "lora": {k: acfg.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules")} if acfg else None,
        "optimizer": "adamw_torch (TRL default; not logged per run)" if cfg else None,
        "checkpoints": sorted(str(p.relative_to(ROOT)) for p in run.glob("checkpoint-*")),
        "evaluations": sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / f"results/**/*{name}*__*.summary.json"), recursive=True))[:60],
        "not_recorded": NOT_RECORDED,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/run_manifest.json")
    a = ap.parse_args()
    tasks = producing_tasks()
    runs = [entry(r, tasks) for r in sorted((ROOT / "runs").iterdir()) if r.is_dir()]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    counts: dict = {}
    for r in runs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    (ROOT / a.out).write_text(json.dumps({"git_commit_at_build": git, "status_counts": counts, "runs": runs}, indent=1))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
