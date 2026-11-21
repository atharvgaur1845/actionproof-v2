"""Build the product's data files from result files only (single source of truth; no number typed by hand).

  product/data/dashboard.json   registered results with intervals, per-seed values, per-class and per-workflow rows,
                                the registration each comes from, and whether it is confirmatory or exploratory
  product/data/replays.json     recorded evaluation episodes: the same fresh task under each model (S3 confirmatory)
    python -m product.build_data
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "product" / "data"
MODELS = {"warm_start": "Warm start (no RL)", "a12_ra_s0": "GRPO, acknowledgement reward (R_A), seed 0",
          "a12_rv_s0": "GRPO, state reward (R_V), seed 0", "s3_matched_s0": "GRPO R_V, class-matched pool, seed 0",
          "s3_rv_mix_s0": "GRPO R_V + recovery-start curriculum (RSC), seed 0"}


def j(p: str):
    return json.loads((ROOT / p).read_text())


def across(d: dict, split: str, k: str) -> dict:
    a = d["splits"][split]["across"][k]
    return {"per_seed": a["per_seed"], "est": a["seed_avg"]["est"], "ci95": a["seed_avg"]["ci95"],
            "p_one_sided": a["sign_flip_p_one_sided"], "n_seeds": len(a["per_seed"])}


def dashboard() -> dict:
    s3, a1, a1pd = j("results/s3/s3_summary.json"), j("results/s3/s3_addendum.json"), j("results/s3/s3_addendum_full_pd.json")
    a12 = j("results/s2/a12_summary.json")
    rows = []
    for sp, name in (("s3_conf_rec", "Fresh recovery starts"), ("s3_conf_full", "Fresh full episodes"),
                     ("s3_conf_stress", "Duplicate / lagging-read stress")):
        for k in ("vsc", "false_success", "violation"):
            rows.append({"comparison": "RSC vs A12 R_V (recipe: start states + fault coverage)", "split": name,
                         "metric": k, **across(s3, sp, k), "registration": "PREREGISTRATION_S3.md",
                         "status": "confirmatory" if k == "vsc" and sp != "s3_conf_stress" else "secondary"})
            rows.append({"comparison": "RSC vs class-matched control (start states only)", "split": name, "metric": k,
                         **across(a1, sp, k), "registration": "PREREGISTRATION_S3_ADDENDUM1.md",
                         "status": "confirmatory" if k == "vsc" and sp == "s3_conf_rec" else "secondary"})
    for k in ("vsc", "false_success", "violation"):
        rows.append({"comparison": "RSC vs class-matched control (start states only)",
                     "split": "Full episodes: partial + duplicate classes", "metric": k, **across(a1pd, "s3_conf_full", k),
                     "registration": "PREREGISTRATION_S3_ADDENDUM1.md", "status": "confirmatory" if k == "vsc" else "secondary"})
    models = {}
    for sp in ("s3_conf_rec", "s3_conf_full", "s3_conf_stress"):
        for t, m in s3["splits"][sp]["models"].items():
            models.setdefault(t, {})[sp] = m
        for t, m in a1["splits"][sp]["models"].items():
            models.setdefault(t, {})[sp] = m
    return {"rows": rows, "models": models, "by_workflow": j("results/s3/s3_by_workflow.json"),
            "a12": {"phantom_readback_per_seed": a12["phantom_readback"]["per_seed"],
                    "vsc_per_seed": {k: v["diff"] for k, v in a12["vsc"]["per_seed"].items()}},
            "limits": ["RSC vs A12 R_V compares recipes (start states and fault coverage); the start-state comparison is "
                       "RSC vs the class-matched control, 3 seeds (exact sign-flip resolution 1/8).",
                       "Against the class-matched control RSC has more violations (+0.027 on the partial and duplicate "
                       "subset) and no gain on duplicate stress.",
                       "No positive transfer to the OpsSim domains (IT operations, healthcare administration).",
                       "One model family (Tiny Aya Global 3.35B, LoRA) and one authored simulator."],
            "sources": {"S3": "results/s3/s3_summary.json", "S3 addendum": "results/s3/s3_addendum.json",
                        "S3 addendum subset": "results/s3/s3_addendum_full_pd.json", "A12": "results/s2/a12_summary.json"}}


def replays(n_per_class: int = 2) -> dict:
    eps = {}
    for t in MODELS:
        p = ROOT / f"results/s3/conf/{t}__s3_conf_full.jsonl"
        eps[t] = {r["task_id"]: r for r in map(json.loads, p.open())}
    common = sorted(set.intersection(*[set(v) for v in eps.values()]))
    by_class: dict[str, list[str]] = {}
    for tid in common:
        by_class.setdefault(eps["warm_start"][tid]["fault_class"], []).append(tid)
    chosen = [tid for c in sorted(by_class) for tid in by_class[c][:n_per_class]]
    out = {"models": MODELS, "tasks": []}
    for tid in chosen:
        t0 = eps["warm_start"][tid]
        out["tasks"].append({"task_id": tid, "fault_class": t0["fault_class"], "workflow": t0["task"]["workflow"],
                             "prompt": t0["task"]["prompt"], "task": t0["task"], "runs": {
                                 m: {"trace": eps[m][tid]["trace"], "report": eps[m][tid]["report"],
                                     "score": {k: eps[m][tid]["score"][k] for k in ("vsc", "goal_met", "false_success",
                                                                                     "duplicate_irreversible", "collateral", "steps")},
                                     "violation": eps[m][tid]["violation"], "policy_calls": eps[m][tid]["policy_calls"]}
                                 for m in MODELS}})
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dashboard.json").write_text(json.dumps(dashboard(), indent=1))
    r = replays()
    (OUT / "replays.json").write_text(json.dumps(r))
    print("dashboard rows", len(dashboard()["rows"]), "replay tasks", len(r["tasks"]))
