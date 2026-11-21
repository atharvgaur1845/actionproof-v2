"""FCA development verdict (docs/research/algorithm/FCA_GRPO.md §7, criteria fixed before training).

Against recovery-start GRPO (`s3_rv_mix`, same pool, steps, rollouts and seeds 0-1):
  kill 1  s3_rec_dev VSC not at least +0.03 above, averaged over seeds 0-1
  kill 2  proc_dev VSC falls by more than 0.03
  kill 3  KL at step 100 above 10x recovery-start GRPO's
Also: per-class dev VSC, and fork-group diagnostics from the training logs (share of fork groups whose on-policy
rewards have zero variance; share in which some forced class's Q differs from the on-policy mean V).
    python scripts/s4_fca_dev.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1)


def summ(tag: str, split: str) -> dict:
    d = "results/s4/dev" if tag.startswith("s4_fca") else "results/s3/dev"
    return json.loads((ROOT / d / f"{tag}__{split}.summary.json").read_text())


def kl(tag: str, step: int) -> float | None:
    h = json.loads((ROOT / "results/s4/train_state" / f"{tag}.json").read_text())["log_history"]
    return next((x["kl"] for x in h if x.get("step") == step and "kl" in x), None)


def forks(tag: str) -> dict:
    gs = [g for line in (ROOT / "results/s4/train_state" / f"{tag}.fca_groups.jsonl").open()
          for g in json.loads(line)["groups"]]
    informative = [g for g in gs if any(abs(q - g["V"]) > 1e-9 for q in g["Q"].values())]
    return {"fork_groups": len(gs), "on_policy_zero_var_share": sum(g["on_zero_var"] for g in gs) / len(gs),
            "forced_informative_share": len(informative) / len(gs)}


def main() -> None:
    out: dict = {"per_seed": {}, "by_class_s3_rec_dev": {}}
    for s in SEEDS:
        f, r = f"s4_fca_s{s}", f"s3_rv_mix_s{s}"
        out["per_seed"][s] = {sp: {"fca": summ(f, sp)["vsc"], "rsc": summ(r, sp)["vsc"],
                                   "fca_false_success": summ(f, sp)["false_success"],
                                   "rsc_false_success": summ(r, sp)["false_success"]}
                              for sp in ("s3_rec_dev", "proc_dev")}
        out["per_seed"][s]["kl_step100"] = {"fca": kl(f, 100), "rsc": kl(r, 100)}
        out["per_seed"][s]["forks"] = forks(f)
    d_rec = sum(v["s3_rec_dev"]["fca"] - v["s3_rec_dev"]["rsc"] for v in out["per_seed"].values()) / len(SEEDS)
    d_proc = sum(v["proc_dev"]["fca"] - v["proc_dev"]["rsc"] for v in out["per_seed"].values()) / len(SEEDS)
    kl_ratio = max(v["kl_step100"]["fca"] / v["kl_step100"]["rsc"] for v in out["per_seed"].values())
    out["criteria"] = {"s3_rec_dev_diff": d_rec, "kill_1": d_rec < 0.03,
                       "proc_dev_diff": d_proc, "kill_2": d_proc < -0.03,
                       "kl_ratio_step100_max": kl_ratio, "kill_3": kl_ratio > 10}
    out["verdict"] = "dropped" if any(out["criteria"][k] for k in ("kill_1", "kill_2", "kill_3")) else "survives"
    for c in summ("s4_fca_s0", "s3_rec_dev")["by_class"]:
        out["by_class_s3_rec_dev"][c] = {
            "n": summ("s4_fca_s0", "s3_rec_dev")["by_class"][c]["n"],
            "fca": [summ(f"s4_fca_s{s}", "s3_rec_dev")["by_class"][c]["vsc"] for s in SEEDS],
            "rsc": [summ(f"s3_rv_mix_s{s}", "s3_rec_dev")["by_class"][c]["vsc"] for s in SEEDS]}
    (ROOT / "results/s4/fca_dev_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("criteria", "verdict")}, indent=1))
    for s, v in out["per_seed"].items():
        print(s, v)


if __name__ == "__main__":
    main()
