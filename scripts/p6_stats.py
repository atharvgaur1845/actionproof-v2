"""P6 preregistered paired analysis. Reads P6 outputs in results/p6 and P4 Dw_aug baselines in results/p4."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = 10_000
RNG_SEED = 20261129
SPLITS = ("s4_api", "s4_wf4", "p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
CANONICAL = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
WORKFLOW = ("s4_wf4", "p1_test_wf4")
SEEDS = (0, 1, 2)


def read_model(tag: str, split: str, phase: str) -> dict[str, dict[str, float]]:
    path = ROOT / f"results/{phase}/{tag}__{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen_sample: dict[str, set[int]] = defaultdict(set)
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        tid, sample = str(row["task_id"]), int(row["sample"])
        if sample in seen_sample[tid]:
            raise ValueError(f"duplicate task/sample in {path}:{line_no}: {tid}/{sample}")
        seen_sample[tid].add(sample)
        if sample < 0:  # greedy episode is kept separately
            grouped[tid]["greedy_vsc"].append(float(row["vsc"]))
            continue
        score = row["score"]
        grouped[tid]["vsc"].append(float(row["vsc"]))
        grouped[tid]["false_success"].append(float(score["false_success"]))
        grouped[tid]["violation"].append(float(row["violation"]))
        grouped[tid]["duplicate"].append(float(score["duplicate_irreversible"]))
        grouped[tid]["calls"].append(float(row["policy_calls"]))
        grouped[tid]["reads"].append(float(row["policy_reads"]))
        grouped[tid]["invalid_calls"].append(float(row["invalid"]))
        grouped[tid]["turns"].append(float(row["turns"]))
    out: dict[str, dict[str, float]] = {}
    for tid, metrics in grouped.items():
        if len(metrics["vsc"]) != 4:
            raise ValueError(f"expected 4 sampled episodes for {tag}/{split}/{tid}; got {len(metrics['vsc'])}")
        out[tid] = {k: float(np.mean(v)) for k, v in metrics.items()}
    return out


def load_arm(prefix: str, phase: str, split: str) -> list[dict[str, dict[str, float]]]:
    return [read_model(f"{prefix}_s{s}", split, phase) for s in SEEDS]


def merge_splits(models: list[dict[str, dict[str, float]]], splits: tuple[str, ...],
                 prefix: str, phase: str) -> list[dict[str, dict[str, float]]]:
    merged = [dict() for _ in SEEDS]
    for split in splits:
        current = load_arm(prefix, phase, split)
        for k in range(3):
            overlap = set(merged[k]) & set(current[k])
            if overlap:
                raise ValueError(f"task IDs overlap across splits: {sorted(overlap)[:3]}")
            merged[k].update(current[k])
    return merged


def bootstrap_contrast(cand: list[dict], base: list[dict], metric: str, rng: np.random.Generator) -> dict:
    tasks = sorted(set.intersection(*[set(x) for x in cand + base]))
    if not tasks:
        raise ValueError("no common task IDs")
    if any(set(x) != set(tasks) for x in cand + base):
        raise ValueError("task sets differ across models; refusing a partial comparison")
    seed_deltas = [float(np.mean([x[t][metric] - y[t][metric] for t in tasks]))
                   for x, y in zip(cand, base)]
    per_task = np.asarray([np.mean([cand[s][t][metric] for s in range(3)]) -
                           np.mean([base[s][t][metric] for s in range(3)]) for t in tasks], dtype=float)
    idx = rng.integers(0, len(tasks), size=(B, len(tasks)))
    draws = per_task[idx].mean(axis=1)
    est = float(per_task.mean())
    return {"metric": metric, "n_tasks": len(tasks), "estimate": est,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "p_one_sided": float((np.count_nonzero(draws <= 0) + 1) / (B + 1)),
            "seed_deltas": seed_deltas}


def test_ni(result: dict, margin: float = -0.03) -> bool:
    return result["ci95"][0] > margin and all(x > margin for x in result["seed_deltas"])


def arm_means(models: list[dict[str, dict[str, float]]]) -> dict[str, float]:
    keys = sorted(set.intersection(*[set(m) for m in models]))
    metrics = sorted(set.intersection(*[set(m[t]) for m in models for t in keys]))
    return {k: float(np.mean([m[t][k] for m in models for t in keys])) for k in metrics}


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    out: dict = {"registration": "PREREGISTRATION_P6.md", "bootstrap_replicates": B,
                 "bootstrap_seed": RNG_SEED, "splits": {}, "hypotheses": {}}

    # Split-level comparisons: P6-SURF vs P6-CAN and P6-SURF vs the frozen P4 Dw_aug SFT reference.
    for split in SPLITS:
        surf = load_arm("P6S", "p6", split)
        can = load_arm("P6C", "p6", split)
        sft = load_arm("Dwaug", "p4", split)
        out["splits"][split] = {
            "means": {"P6_SURF": arm_means(surf), "P6_CAN": arm_means(can), "Dw_aug_SFT": arm_means(sft)},
            "surface_vs_canonical": bootstrap_contrast(surf, can, "vsc", rng),
            "surface_vs_dwaug_sft": bootstrap_contrast(surf, sft, "vsc", rng),
            "false_success_surface_vs_sft": bootstrap_contrast(surf, sft, "false_success", rng),
            "violation_surface_vs_sft": bootstrap_contrast(surf, sft, "violation", rng),
            "duplicates_surface_vs_sft": bootstrap_contrast(surf, sft, "duplicate", rng),
            "calls_surface_vs_sft": bootstrap_contrast(surf, sft, "calls", rng),
            "reads_surface_vs_sft": bootstrap_contrast(surf, sft, "reads", rng),
            "invalid_calls_surface_vs_sft": bootstrap_contrast(surf, sft, "invalid_calls", rng),
            "turns_surface_vs_sft": bootstrap_contrast(surf, sft, "turns", rng),
        }

    h1 = out["splits"]["s4_api"]["surface_vs_canonical"]
    h2 = out["splits"]["s4_api"]["surface_vs_dwaug_sft"]
    # Merge the split families with matched seed models; task IDs must be unique across constituent splits.
    surf_work = merge_splits([], WORKFLOW, "P6S", "p6")
    sft_work = merge_splits([], WORKFLOW, "Dwaug", "p4")
    surf_can = merge_splits([], CANONICAL, "P6S", "p6")
    sft_can = merge_splits([], CANONICAL, "Dwaug", "p4")
    h3 = bootstrap_contrast(surf_work, sft_work, "vsc", rng)
    h4 = bootstrap_contrast(surf_can, sft_can, "vsc", rng)
    fcr = bootstrap_contrast(surf_can, sft_can, "false_success", rng)

    out["hypotheses"] = {
        "H1_surface_randomisation_improves_heldout_api": {
            **h1, "supported": bool(h1["p_one_sided"] < 0.05 and h1["ci95"][0] > 0 and all(x > 0 for x in h1["seed_deltas"]))},
        "H2_surface_randomised_rl_retains_api_transfer": {**h2, "margin": -0.03, "supported": test_ni(h2)},
        "H3_surface_randomised_rl_retains_unseen_workflow": {**h3, "margin": -0.03, "supported": test_ni(h3)},
        "H4_surface_randomised_rl_retains_canonical_performance": {**h4, "margin": -0.03, "supported": test_ni(h4)},
        "FCR_guardrail_surface_vs_dwaug_canonical": {
            **fcr, "upper_margin": 0.02,
            "passes": bool(fcr["ci95"][1] < 0.02 and all(x < 0.02 for x in fcr["seed_deltas"]))},
    }
    out_path = ROOT / "results/p6/p6_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    for name, result in out["hypotheses"].items():
        print(name, "estimate=", round(result["estimate"], 4), "CI=", [round(x, 4) for x in result["ci95"]],
              "seed_deltas=", [round(x, 4) for x in result["seed_deltas"]],
              "supported/passes=", result.get("supported", result.get("passes")))


if __name__ == "__main__":
    main()
