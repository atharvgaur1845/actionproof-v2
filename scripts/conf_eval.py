"""A12 addendum 2 §2: confirmatory evaluation of one frozen model on one confirmatory split, exactly once.

- The adapter must be in results/a12/frozen_models.json with a matching sha256.
- An access log on the private HF dataset (access/a12_conf_access.jsonl) is checked before starting: a (sha256, split)
  already marked completed is refused; one marked started without completion is resumed, never restarted.
- Episodes are generated with the frozen loop of scripts/eval_proc.py (`run`, imported, not modified), in chunks of
  50 tasks appended to the output, so a crash resumes from the saved per-task output.
    python scripts/conf_eval.py --tag a12_proc_rv_s0 --split proc_conf
Output: results/eval/a12/conf/<tag>__neutral__<split>.jsonl (+ .summary.json), the paths scripts/s2_stats.py reads.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO, LOG = "atharvgaur18/actionproof-jobs", "access/a12_conf_access.jsonl"
SPLITS = ("proc_conf", "proc_combo", "proc_stress")
CHUNK = 50


def _ep():
    spec = importlib.util.spec_from_file_location("eval_proc", ROOT / "scripts/eval_proc.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def access_log(api) -> list[dict]:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO, LOG, repo_type="dataset", force_download=True)
    except Exception:  # noqa: BLE001 - no log yet
        return []
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def append_log(api, entries: list[dict], entry: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("".join(json.dumps(e) + "\n" for e in entries + [entry]))
    api.upload_file(path_or_fileobj=f.name, path_in_repo=LOG, repo_id=REPO, repo_type="dataset",
                    commit_message=f"conf access {entry['tag']} {entry['split']} {entry['event']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--model", default="runs/rft_proc/merged")
    a = ap.parse_args()
    frozen = {}
    for name in ("frozen_models.json", "frozen_models_later.json"):  # the second holds models frozen after A14's freeze
        f = ROOT / "results/a12" / name
        if f.exists():
            frozen.update(json.loads(f.read_text()))
    frozen = frozen[a.tag]
    adapter = ROOT / frozen["adapter"]
    h = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    if h != frozen["sha256"]:
        raise SystemExit(f"{a.tag}: adapter sha256 {h} differs from the frozen {frozen['sha256']}")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    log = access_log(api)
    mine = [e for e in log if e["sha256"] == h and e["split"] == a.split]
    if any(e["event"] == "completed" for e in mine):
        raise SystemExit(f"{a.tag} {a.split}: already completed; re-runs are forbidden")
    out = ROOT / f"results/eval/a12/conf/{a.tag}__neutral__{a.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["task_id"] for line in out.open()} if out.exists() else set()
    if mine and not done:
        print(f"{a.tag} {a.split}: started before with no saved output; continuing from the first task")
    append_log(api, log, {"tag": a.tag, "sha256": h, "split": a.split, "event": "resumed" if mine else "started",
                          "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks_already_saved": len(done)})

    ep = _ep()
    from vllm import LLM
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.sim.prompts import PROC_PROMPTS
    from actionproof.train.merge import merge
    llm = LLM(merge(str(adapter), a.model), gpu_memory_utilization=0.85, max_model_len=8192, dtype="bfloat16", seed=0)
    tasks = [t for t in load_tasks(ROOT / f"data/bench/proc/{a.split}.jsonl") if t.task_id not in done]
    t0 = time.time()
    for i in range(0, len(tasks), CHUNK):
        recs = ep.run(llm, tasks[i:i + CHUNK], PROC_PROMPTS["neutral"], 1, 0.0, 16, 256, 0)
        with out.open("a") as f:
            for r in recs:
                f.write(json.dumps({**r, "model": a.model, "adapter": frozen["adapter"], "adapter_sha256": h,
                                    "prompt": "neutral"}) + "\n")
    recs = [json.loads(line) for line in out.open()]
    s = {**ep.summarize(recs), "tag": a.tag, "adapter_sha256": h, "split": a.split, "seconds": round(time.time() - t0)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    append_log(api, access_log(api), {"tag": a.tag, "sha256": h, "split": a.split, "event": "completed",
                                      "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(recs)})
    print(json.dumps({k: s[k] for k in ("n", "vsc", "phantom_readback", "false_success", "seconds")}))


if __name__ == "__main__":
    main()
