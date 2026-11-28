"""Freeze P6 source, training/evaluation data, Dw_aug adapters and pre-existing P4 baseline outputs.

Run `python scripts/freeze_p6.py --register` once after installing this patch and before any P6 training.
Commit the generated results/p6_hashes.json. Every P6 train/eval task runs `--check` before doing work.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "PREREGISTRATION_P6.md", "scripts/freeze_p6.py", "scripts/p6_eval.py", "scripts/p6_stats.py",
    "scripts/p4_eval.py", "scripts/p4_stats.py", "scripts/p1_eval.py", "scripts/freeze_p4.py",
    "actionproof/stage3/train.py", "actionproof/stage3/rollout.py", "actionproof/stage3/env.py",
    "actionproof/stage4/train_variants.py", "actionproof/stage4/surface.py", "actionproof/stage4/shift.py",
    "actionproof/stage3/rewards.py", "actionproof/integrity/sim/truth.py",
    "actionproof/integrity/sim/runtime.py", "actionproof/integrity/sim/prompts.py",
    "actionproof/rollout/episode.py", "actionproof/rollout/agenttune_rollout.py",
    "actionproof/train/merge.py", "results/p4_hashes.json",
    "slurm/p6_train_common.sh", "slurm/p6_eval_common.sh",
    "slurm/tasks/2300_p6_can_s0.sh", "slurm/tasks/2300_p6_can_s1.sh", "slurm/tasks/2300_p6_can_s2.sh",
    "slurm/tasks/2300_p6_surf_s0.sh", "slurm/tasks/2300_p6_surf_s1.sh", "slurm/tasks/2300_p6_surf_s2.sh",
    "slurm/tasks/2310_p6_eval_can_s0.sh", "slurm/tasks/2310_p6_eval_can_s1.sh", "slurm/tasks/2310_p6_eval_can_s2.sh",
    "slurm/tasks/2310_p6_eval_surf_s0.sh", "slurm/tasks/2310_p6_eval_surf_s1.sh", "slurm/tasks/2310_p6_eval_surf_s2.sh",
    "slurm/p6_hfjob_run.sh", "scripts/p6_hf_launch.sh",
    "slurm/tasks/2320_p6_stats.sh", "tests/test_p6_stats.py",
]
DATA = [
    "data/bench/proc/s3_trainb_half.jsonl", "data/bench/proc/s3_rec_train_half.jsonl",
    "data/bench/proc/s4_api.jsonl", "data/bench/proc/s4_wf4.jsonl",
    "data/bench/proc/p1_test_full.jsonl", "data/bench/proc/p1_test_pp.jsonl",
    "data/bench/proc/p1_test_rec.jsonl", "data/bench/proc/p1_test_wf4.jsonl",
    "data/bench/proc/p1_test_combo.jsonl",
]
SPLITS = ("s4_api", "s4_wf4", "p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
BASELINES = [f"results/p4/Dwaug_s{s}__{sp}.jsonl" for s in range(3) for sp in SPLITS]
ADAPTERS = [f"runs/p4_dwaug_s{s}/final" for s in range(3)]
MANIFEST = ROOT / "results/p6_hashes.json"


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_file(rel: str) -> str:
    p = ROOT / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    return file_hash(p)


def snapshot_tree(rel: str) -> dict:
    root = ROOT / rel
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"empty adapter directory: {root}")
    rows = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}
    blob = "\n".join(f"{k} {v}" for k, v in rows.items()).encode()
    return {"sha256": hashlib.sha256(blob).hexdigest(), "files": rows}


def current() -> dict:
    return {
        "files": {p: snapshot_file(p) for p in FILES},
        "data": {p: snapshot_file(p) for p in DATA},
        "p4_baseline_outputs": {p: snapshot_file(p) for p in BASELINES},
        "p4_dwaug_adapters": {p: snapshot_tree(p) for p in ADAPTERS},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--register", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.register:
        if MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} already exists; refusing to overwrite registration")
        snapshot = current()
        payload = {"registered_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "registration": "PREREGISTRATION_P6.md", **snapshot}
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"registered P6 snapshot: {MANIFEST}")
        return
    if not MANIFEST.is_file():
        raise SystemExit("P6 has not been registered; run --register before training")
    registered = json.loads(MANIFEST.read_text())
    want = {k: registered[k] for k in ("files", "data", "p4_baseline_outputs", "p4_dwaug_adapters")}
    got = current()
    if got != want:
        for section in want:
            expected, actual = want[section], got[section]
            bad = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
            if bad:
                print(section + " mismatch:")
                for k in bad[:30]:
                    print("  " + k)
        raise SystemExit(1)
    print("OK: P6 source, data, P4 baselines and initial adapters match the registration")


if __name__ == "__main__":
    main()
