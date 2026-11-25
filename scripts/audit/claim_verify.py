"""Independent re-computation of headline claims from raw episode files (research_audit/CLAIM_VERIFICATION.md).

Run on Slurm (CPU):  python scripts/audit/claim_verify.py --out results/audit/claim_verify.json

Separate pathway. For every episode the deterministic verdicts are recomputed by `scripts/reference_verifier_proc.py`,
which was written from docs/stage2/VERIFIER_SPEC.md alone and imports no simulator or verifier code. It replays the
recorded attempts on the task's initial state. Headline aggregates are then recomputed here with plain means: no code
is shared with the original statistics scripts. Bootstrap intervals and permutation tests are NOT recomputed; only
point estimates and per-seed values are.

Two reference pathways: `replay` (state rebuilt from attempts; its spec, proc1, assumes a partial commit stores
exactly half, so it disagrees on Stage 3/P1 tasks with other partial fractions) and `final_state` (the reference
verdict logic applied to the recorded final database state).

Integrity checks per file:
- record count;
- duplicate (task_id, sample) pairs;
- task ids missing from, or foreign to, the frozen split file;
- agreement of the recorded VSC with the reference verifier's VSC.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location("refv", ROOT / "scripts/reference_verifier_proc.py")
REF = importlib.util.module_from_spec(_s)
_s.loader.exec_module(REF)


def split_ids(split: str) -> set | None:
    p = ROOT / f"data/bench/proc/{split}.jsonl"
    return {json.loads(l)["task_id"] for l in p.read_text().splitlines()} if p.exists() else None


def check_file(path: str, split: str, use_ref: bool) -> dict:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    keys = [(r.get("task_id"), r.get("sample", -1)) for r in rows]
    ids = {k[0] for k in keys}
    sid = split_ids(split)
    out = {"file": str(Path(path).relative_to(ROOT)), "records": len(rows), "duplicates": len(keys) - len(set(keys)),
           "tasks": len(ids), "split_tasks": len(sid) if sid else None,
           "missing_from_file": len(sid - ids) if sid else None, "foreign_to_split": len(ids - sid) if sid else None}
    rec, ref, agree, refrb_n, refrb_d, warn = [], [], 0, 0, 0, 0
    per = {"greedy": defaultdict(lambda: {"rec": [], "ref": []}), "sampled": defaultdict(lambda: {"rec": [], "ref": []})}
    for r in rows:
        v = bool(r.get("vsc", (r.get("score") or {}).get("vsc")))
        rec.append(v)
        mode = "greedy" if (r.get("sample") is None or r.get("sample", -1) < 0) else "sampled"
        per[mode][r["task_id"]]["rec"].append(v)
        if use_ref:
            try:
                res = REF.verify_episode(r)
                rv = bool(res["vsc"])
                fs = REF.verify_episode(r, source="final_state")
                fsv = bool(fs["vsc"])
                out["fs_agree"] = out.get("fs_agree", 0) + (fsv == v)
                out["replay_state_mismatch"] = out.get("replay_state_mismatch", 0) + (res["state_match"] is False)
                per[mode][r["task_id"]].setdefault("fs", []).append(fsv)
                m = res["metrics"]
                refrb_n += m["phantom_rb_num"]
                refrb_d += m["phantom_rb_den"]
                warn += bool(res["warnings"])
            except Exception as e:  # recorded, never hidden
                rv = None
                out.setdefault("ref_errors", []).append(f"{r.get('task_id')}: {type(e).__name__}: {e}"[:160])
            ref.append(rv)
            per[mode][r["task_id"]]["ref"].append(rv)
            agree += rv == v
    out["rec_vsc"] = sum(rec) / len(rec) if rec else None
    if use_ref:
        ok = [x for x in ref if x is not None]
        out["fs_agreement"] = out.pop("fs_agree", 0) / len(rows) if rows else None
        out.update(ref_vsc=sum(ok) / len(ok) if ok else None, agreement=agree / len(rows) if rows else None,
                   ref_phantom_readback=(refrb_n / refrb_d) if refrb_d else None, ref_phantom_den=refrb_d,
                   episodes_with_ref_warnings=warn)
    # task-level means (each task weighs equally, samples averaged within task), as every registered statistic does
    def tm(P, k):
        return (sum(sum(x for x in d[k] if x is not None) / max(1, len([x for x in d[k] if x is not None]))
                    for d in P.values()) / len(P)) if P else None
    for mode, P in per.items():
        out[f"rec_task_mean_{mode}"] = tm(P, "rec")
        if use_ref:
            out[f"ref_task_mean_{mode}"] = tm(P, "ref")
            out[f"fs_task_mean_{mode}"] = tm({k: v for k, v in P.items() if "fs" in v}, "fs")
    return out


def group(pattern: str, split_of, use_ref: bool, rx: str) -> dict:
    res = {}
    for f in sorted(glob.glob(str(ROOT / pattern))):
        m = re.search(rx, Path(f).name)
        if not m or Path(f).name.endswith(".summary.json"):
            continue
        res[Path(f).name] = {"arm": m.group("arm"), "seed": int(m.group("seed")), **check_file(f, split_of(Path(f).name), use_ref)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    out["A12_proc_conf"] = group("results/eval/a12/conf/a12_*_s*__neutral__proc_conf.jsonl", lambda n: "proc_conf", True,
                                 r"a12_(?P<arm>.+)_s(?P<seed>\d)__neutral__proc_conf\.jsonl")
    for sp in ("s3_conf_rec", "s3_conf_full"):
        out[f"S3_{sp}"] = group(f"results/s3/conf/*__{sp}.jsonl", lambda n, sp=sp: sp, True,
                                rf"(?P<arm>.+)_s(?P<seed>\d)__{sp}\.jsonl")
    p1 = lambda n: n.split("__")[1].replace(".jsonl", "")
    out["P1_test"] = group("results/p1/test/*_s[0-9]__p1_test_*.jsonl", p1, True, r"(?P<arm>[A-D])_s(?P<seed>\d)__p1_test_\w+\.jsonl")
    out["P2_test"] = group("results/p2/test/*_s[0-9]__p1_test_*.jsonl", p1, True, r"(?P<arm>[A-Za-z0-9]+)_s(?P<seed>\d)__p1_test_\w+\.jsonl")
    out["AD_C1_cw_conf"] = group("results/ad/conf/ad_*_s[0-9]__cw_conf.jsonl", lambda n: "cw_conf", False,
                                 r"ad_(?P<arm>.+)_s(?P<seed>\d)__cw_conf\.jsonl")
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(out, indent=1, default=str))
    for k, g in out.items():
        n = len(g)
        dup = sum(v["duplicates"] for v in g.values())
        ag = [v["agreement"] for v in g.values() if v.get("agreement") is not None]
        print(k, "files", n, "dups", dup, "min agreement", min(ag) if ag else None)


if __name__ == "__main__":
    main()
