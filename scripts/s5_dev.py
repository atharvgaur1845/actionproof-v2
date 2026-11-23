"""Stage 5 development verdicts (docs/research/algorithm/C5G_DEV.md §4; criteria fixed before training).

For each candidate (c5, guide), against recovery-start GRPO (`s3_rv_mix`, same pool, steps, rollouts, seeds 0-1;
development numbers already on file in results/s3/dev and results/s4/train_state):
  kill 1  s3_rec_dev VSC not at least +0.03 above RSC, averaged over seeds 0-1
  kill 2  proc_dev VSC falls by more than 0.03
  kill 3  KL at step 100 more than 10x RSC's (either seed)
Descriptive: per-class s3_rec_dev VSC; from the training rollout logs, the share of partial-payment episodes that
refund and then re-pay with a new key, by 50-step block; hinted-episode share (guide).
    python scripts/s5_dev.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1)
MODES = ("c5", "guide")


def summ(tag: str, split: str) -> dict:
    d = "results/s5/dev" if tag.startswith("s5_") else "results/s3/dev"
    return json.loads((ROOT / d / f"{tag}__{split}.summary.json").read_text())


def kl(tag: str, step: int) -> float | None:
    d = "results/s5/train_state" if tag.startswith("s5_") else "results/s4/train_state"
    h = json.loads((ROOT / d / f"{tag}.json").read_text())
    h = h["log_history"] if isinstance(h, dict) else h
    return next((x["kl"] for x in h if x.get("step") == step and "kl" in x), None)


def chain(tag: str) -> dict:
    p = ROOT / f"results/s5/rollouts/{tag}.jsonl.gz"
    if not p.exists():
        return {}
    blocks: dict[int, list[int]] = {}
    hinted = n = 0
    for line in gzip.open(p, "rt"):
        r = json.loads(line)
        n += 1
        hinted += bool(r.get("hinted"))
        f = (r["task"]["meta"].get("faults") or [{}])[0]
        if f.get("kind") != "partial" or f.get("tool") != "pay_supplier":
            continue
        tr = [x for x in r["trace"] if x.get("by", "agent") == "agent"]
        keys = [(x.get("args") or {}).get("idempotency_key") for x in tr if x["tool"] == "pay_supplier"]
        refunded = any(x["tool"] == "refund_payment" for x in tr)
        b = blocks.setdefault((r["step"] or 0) // 50, [0, 0])
        b[0] += 1
        b[1] += refunded and len(keys) >= 1 and len(set(keys)) == len(keys) and r["score"]["steps"].get("pay") == "VERIFIED_SUCCESS"
    return {"episodes": n, "hinted_share": hinted / max(n, 1),
            "partial_pay_repaired_by_block": {str(k * 50): v[1] / v[0] for k, v in sorted(blocks.items())}}


def main() -> None:
    out: dict = {}
    for m in MODES:
        res: dict = {"per_seed": {}, "by_class_s3_rec_dev": {}}
        try:
            for s in SEEDS:
                c, r = f"s5_{m}_s{s}", f"s3_rv_mix_s{s}"
                res["per_seed"][s] = {sp: {"cand": summ(c, sp)["vsc"], "rsc": summ(r, sp)["vsc"],
                                           "cand_false_success": summ(c, sp)["false_success"],
                                           "rsc_false_success": summ(r, sp)["false_success"],
                                           "cand_calls": summ(c, sp)["policy_calls"], "rsc_calls": summ(r, sp)["policy_calls"]}
                                      for sp in ("s3_rec_dev", "proc_dev")}
                res["per_seed"][s]["kl_step100"] = {"cand": kl(c, 100), "rsc": kl(r, 100)}
                res["per_seed"][s]["training"] = chain(c)
        except FileNotFoundError as e:
            out[m] = {"missing": str(e)}
            continue
        v = res["per_seed"].values()
        d_rec = sum(x["s3_rec_dev"]["cand"] - x["s3_rec_dev"]["rsc"] for x in v) / len(SEEDS)
        d_proc = sum(x["proc_dev"]["cand"] - x["proc_dev"]["rsc"] for x in v) / len(SEEDS)
        ratio = max(x["kl_step100"]["cand"] / x["kl_step100"]["rsc"] for x in v)
        res["criteria"] = {"s3_rec_dev_diff": d_rec, "kill_1": d_rec < 0.03, "proc_dev_diff": d_proc,
                           "kill_2": d_proc < -0.03, "kl_ratio_step100_max": ratio, "kill_3": ratio > 10}
        res["verdict"] = "dropped" if any(res["criteria"][k] for k in ("kill_1", "kill_2", "kill_3")) else "survives"
        for cl in summ(f"s5_{m}_s0", "s3_rec_dev")["by_class"]:
            res["by_class_s3_rec_dev"][cl] = {
                "cand": [summ(f"s5_{m}_s{s}", "s3_rec_dev")["by_class"].get(cl, {}).get("vsc") for s in SEEDS],
                "rsc": [summ(f"s3_rv_mix_s{s}", "s3_rec_dev")["by_class"].get(cl, {}).get("vsc") for s in SEEDS]}
        out[m] = res
        print(m, json.dumps({k: res[k] for k in ("criteria", "verdict")}))
    (ROOT / "results/s5/s5_dev_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results/s5/s5_dev_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
