"""Descriptive S3 numbers from the post-results review (docs/stage3/ACADEMIC_REVIEW_POST.md), recomputed from the
per-model files so the paper and README can cite them through the claims ledger. Not registered analyses.

  transfer      care_admin and it_ops VSC and collateral per arm (warm start, A12 R_V s0-4, RSC s0-4, control s0-2)
  rp_cell       partial payments after a goods receipt (receive_pay, faulted pay_supplier) on s3_conf_full and
                s3_conf_rec: tasks repaired (VSC 1) per model
  learned       partial-class VSC on s3_conf_full per run; runs at or above 0.40
  clean         calls and reads per episode on clean s3_conf_full tasks, per arm
  same_calls    s3_conf_full tasks on which RSC seed 0 and control seed 0 make identical tool calls
    python scripts/s3_review_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "results/s3/conf"
ARMS = {"warm_start": ["warm_start"], "a12_rv": [f"a12_rv_s{s}" for s in range(5)],
        "rsc": [f"s3_rv_mix_s{s}" for s in range(5)], "control": [f"s3_matched_s{s}" for s in range(3)]}


def summ(tag: str, split: str) -> dict:
    return json.loads((C / f"{tag}__{split}.summary.json").read_text())


def rows(tag: str, split: str):
    return [json.loads(line) for line in (C / f"{tag}__{split}.jsonl").open()]


def main() -> None:
    out: dict = {"transfer": {}, "rp_cell": {}, "learned": {}, "clean": {}}
    for d in ("care_admin", "it_ops"):
        T = {}
        for arm, tags in ARMS.items():
            v = [summ(t, f"s3_{d}")["vsc"] for t in tags]
            c = [summ(t, f"s3_{d}")["collateral"] for t in tags]
            T[arm] = {"vsc": v, "vsc_mean": sum(v) / len(v), "collateral": c, "collateral_mean": sum(c) / len(c)}
        T["rsc_below_rv_vsc_pairs"] = sum(r < a for r, a in zip(T["rsc"]["vsc"], T["a12_rv"]["vsc"]))
        T["rsc_above_rv_collateral_pairs"] = sum(r > a for r, a in zip(T["rsc"]["collateral"], T["a12_rv"]["collateral"]))
        out["transfer"][d] = T
    for sp in ("s3_conf_full", "s3_conf_rec"):
        cell = {}
        for arm, tags in ARMS.items():
            per = []
            for t in tags:
                rs = [r for r in rows(t, sp) if r["task"]["workflow"] == "receive_pay"
                      and (r["task"]["meta"].get("faults") or [{}])[0].get("kind") == "partial"
                      and (r["task"]["meta"].get("faults") or [{}])[0].get("tool") == "pay_supplier"]
                per.append({"n": len(rs), "repaired": sum(int(r["vsc"]) for r in rs)})
            cell[arm] = {"per_model": per, "repaired": sum(p["repaired"] for p in per), "n": sum(p["n"] for p in per)}
        out["rp_cell"][sp] = cell
    for arm, tags in ARMS.items():
        v = [summ(t, "s3_conf_full")["by_class"]["partial"]["vsc"] for t in tags]
        out["learned"][arm] = {"partial_vsc": v, "runs_ge_040": sum(x >= 0.40 for x in v), "runs": len(v)}
        cl = [summ(t, "s3_conf_full")["by_class"]["clean"] for t in tags]
        out["clean"][arm] = {"calls": sum(x["policy_calls"] for x in cl) / len(cl),
                             "reads": sum(x["policy_reads"] for x in cl) / len(cl)}
    def seqs(tag):
        return {r["task_id"]: [(e["tool"], {k: v for k, v in (e.get("args") or {}).items() if k != "summary"})
                               for e in r["trace"] if e.get("by") == "agent"] for r in rows(tag, "s3_conf_full")}
    a, b = seqs("s3_rv_mix_s0"), seqs("s3_matched_s0")
    out["same_calls"] = {"rsc_s0_vs_control_s0": sum(a[t] == b[t] for t in a), "n": len(a)}
    (ROOT / "results/s3/s3_review_numbers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({d: {a: (round(v["vsc_mean"], 3), round(v["collateral_mean"], 3)) if isinstance(v, dict) else v
                          for a, v in T.items()} for d, T in out["transfer"].items()}, indent=1))
    print({sp: {a: (c["repaired"], c["n"]) for a, c in cell.items()} for sp, cell in out["rp_cell"].items()})
    print({a: (v["runs_ge_040"], v["runs"]) for a, v in out["learned"].items()}, out["clean"])


if __name__ == "__main__":
    main()
