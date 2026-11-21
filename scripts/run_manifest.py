"""Run manifest (one entry per trained model): adapter sha256, run configuration, seed, training-pool sha256, steps,
rollouts, the registration it belongs to, and the result files that evaluate it. Run where the adapters live.
    python scripts/run_manifest.py --out results/run_manifest.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"a12_proc_rv": "PREREGISTRATION_A12*.md", "a12_proc_ra": "PREREGISTRATION_A12*.md",
            "k1_": "PREREGISTRATION_K1.md", "s3_rv_mix": "PREREGISTRATION_S3.md", "s3_matched": "PREREGISTRATION_S3_ADDENDUM1.md",
            "s3_p_": "development pilot (docs/stage3/RESEARCH_REPORT.md §6)", "s3_rv_mix25": "development stage B",
            "s4_fca": "docs/research/algorithm/FCA_GRPO.md (development)"}


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/run_manifest.json")
    a = ap.parse_args()
    out = {}
    for run in sorted((ROOT / "runs").iterdir()):
        cfg_p, ad = run / "run_config.json", run / "final" / "adapter_model.safetensors"
        if not ad.exists():
            continue
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        pools = cfg.get("pools") or cfg.get("task_jsonl")
        pool_hash = {p: sha(ROOT / p) for p in pools.split(",")} if isinstance(pools, str) else None
        fam = next((v for k, v in FAMILIES.items() if run.name.startswith(k)), None)
        if run.name.startswith("s3_rv_mix25"):
            fam = FAMILIES["s3_rv_mix25"]
        hist = run / "log_history.json"
        steps = None
        if hist.exists():
            h = json.loads(hist.read_text())
            steps = max((x.get("step", 0) for x in h if isinstance(x, dict)), default=None)
        out[run.name] = {"adapter": str(ad.relative_to(ROOT)), "adapter_sha256": sha(ad), "seed": cfg.get("seed"),
                         "reward": cfg.get("reward", cfg.get("arm")), "algo": cfg.get("algo", "grpo"),
                         "steps": cfg.get("steps"), "logged_last_step": steps,
                         "rollouts_per_step": (cfg.get("num_generations") or 8) * (cfg.get("prompts_per_step") or 4),
                         "lr": cfg.get("lr"), "beta": cfg.get("beta"), "pools": pool_hash, "registration": fam,
                         "results": sorted(glob.glob(str(ROOT / f"results/**/*{run.name.replace('a12_proc_', 'a12_')}*.summary.json"),
                                                     recursive=True))[:40]}
        out[run.name]["results"] = [str(Path(p).relative_to(ROOT)) for p in out[run.name]["results"]]
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(len(out), "runs")


if __name__ == "__main__":
    main()
