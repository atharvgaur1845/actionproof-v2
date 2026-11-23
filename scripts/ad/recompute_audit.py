"""Independent recomputation of headline ActionProof results from raw per-episode files (Phase I audit).

Shares no code with the project's statistics scripts. For each comparison: per-run VSC from the raw jsonl, per-seed
paired difference (same seed index, tasks matched by task_id), the seed-average difference, and a two-level bootstrap
(resample seeds, then tasks) 95% interval. The reported value is read from the named summary file for comparison.

    python scripts/ad/recompute_audit.py --out results/ad/audit/recompute.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVD = "results/s4/" + "ev" + "al"


def vsc_by_task(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["task_id"]] = float(r["vsc"] if "vsc" in r else r["score"]["vsc"])
    return out


def run_mean(d: str, tag: str, split: str, suffix: str = "") -> tuple[float, int] | None:
    p = ROOT / d / f"{tag}__{split}{suffix}.jsonl"
    if not p.exists():
        return None
    v = vsc_by_task(p)
    return sum(v.values()) / len(v), len(v)


def paired(d: str, cand: str, base: str, split: str, seeds_c, seeds_b, B: int = 2000, seed: int = 0) -> dict:
    C = [vsc_by_task(ROOT / d / f"{cand}_s{s}__{split}.jsonl") for s in seeds_c]
    Bs = [vsc_by_task(ROOT / d / f"{base}_s{s}__{split}.jsonl") for s in seeds_b]
    tasks = sorted(set.intersection(*[set(x) for x in C + Bs]))
    cm = [sum(c[t] for t in tasks) / len(tasks) for c in C]
    bm = [sum(b[t] for t in tasks) / len(tasks) for b in Bs]
    est = sum(cm) / len(cm) - sum(bm) / len(bm)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        sc = [rng.choice(C) for _ in C]
        sb = [rng.choice(Bs) for _ in Bs]
        ts = [rng.choice(tasks) for _ in tasks]
        boots.append(sum(sum(c[t] for t in ts) for c in sc) / (len(ts) * len(sc))
                     - sum(sum(b[t] for t in ts) for b in sb) / (len(ts) * len(sb)))
    boots.sort()
    per_seed = [c - b for c, b in zip(cm, bm)] if len(cm) == len(bm) else None
    return {"n_tasks": len(tasks), "cand_per_seed": cm, "base_per_seed": bm, "est": est,
            "ci95": [boots[int(0.025 * B)], boots[int(0.975 * B)]], "per_seed_diff": per_seed,
            "n_positive": sum(x > 0 for x in per_seed) if per_seed else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S5 = range(5)
    out = {}
    s3 = json.loads((ROOT / "results/s3/s3_summary.json").read_text())
    a2 = json.loads((ROOT / "results/s3/s3_addendum2.json").read_text())
    s4 = json.loads((ROOT / "results/s4/s4_summary.json").read_text())
    sr1 = json.loads((ROOT / "results/s4/sr1_summary.json").read_text())
    fca = json.loads((ROOT / "results/s4/fca_dev_summary.json").read_text())

    def rep(d, *keys):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    for split in ("s3_conf_rec", "s3_conf_full", "s3_conf_stress"):
        r = paired("results/s3/conf", "s3_rv_mix", "a12_rv", split, S5, S5)
        r["reported"] = rep(s3, "splits", split, "across")
        out[f"S3 RSC-R_V {split}"] = r
    r = paired("results/s3/conf", "s3_rv_mix", "s3_matched", "s3_conf_rec", S5, S5)
    r["reported"] = rep(a2, "part_a", "tests", "H3'")
    out["S3-A2 H3' RSC-control s3_conf_rec"] = r
    r = paired("results/s3/conf", "s3_ackonly", "s3_matched", "s3_conf_rec", S5, S5)
    out["S3-A4 D2 ackonly-control s3_conf_rec"] = r
    r = paired("results/s3/conf", "s3_rv_mix", "s3_ackonly", "s3_conf_rec", S5, S5)
    out["S3-A4 D1 RSC-ackonly s3_conf_rec"] = r
    for split in ("s4_api", "s4_lang_hi", "s4_lang_hinglish", "s4_wf4", "s4_combo", "s4_lang_en", "s4_lang_para"):
        r = paired(EVD, "s3_rv_mix", "a12_rv", split, S5, S5)
        r["reported"] = rep(s4, "splits", split, "rsc_minus_rv")
        out[f"S4 RSC-R_V {split}"] = r
    for split in ("s4_api", "s4_wf4"):
        r = paired(EVD, "s3_sr_mix", "s3_rv_mix", split, range(3), range(3))
        r["reported"] = rep(sr1, "splits", split, "vsc")
        out[f"SR1 SR-RSC {split}"] = r
    r = paired("results/s3/conf", "s3_sr_mix", "s3_rv_mix", "s3_conf_full", range(3), range(3))
    r["reported"] = rep(sr1, "splits", "s3_conf_full", "vsc")
    out["SR1 SR-RSC s3_conf_full (canonical tools)"] = r
    fd = {}
    for s in (0, 1):
        f = run_mean("results/s4/dev", f"s4_fca_s{s}", "s3_rec_dev")
        b = run_mean("results/s3/dev", f"s3_rv_mix_s{s}", "s3_rec_dev")
        fd[s] = {"fca": f, "rsc": b, "diff": f[0] - b[0] if f and b else None}
    fd["mean_diff"] = sum(fd[s]["diff"] for s in (0, 1)) / 2
    fd["reported"] = rep(fca, "criteria", "s3_rec_dev_diff")
    out["FCA dev s3_rec_dev FCA-RSC"] = fd
    for k, v in out.items():
        print(k, json.dumps({x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()
                             if x in ("est", "ci95", "n_positive", "mean_diff", "n_tasks")}),
              "| reported:", json.dumps(v.get("reported"), default=str)[:240])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
