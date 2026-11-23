"""J1 addendum 1 analysis (PREREGISTRATION_J1_ADDENDUM1.md): v0 (no warning that "ok" is not proof) vs v1, transcript
view, main split, same cases and judges. Writes results/j1/j1_addendum1.json.
    python evaluation/llm_judge_committee/analyze_a1.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JUDGES = ("gemma", "qwen", "gptoss", "deepseek")
UNVERIFIABLE = {("phantom", "noread", 0, "claim"), ("partial", "noread", 0, "claim"), ("never_commit", "noread", 0, "claim")}


def load(j: str, v: str) -> dict:
    f = HERE / "outputs" / f"{j}__transcript__{v}__main.jsonl"
    return {r["case_id"]: r for r in map(json.loads, f.read_text().splitlines())} if f.exists() else {}


def main() -> None:
    import analyze as A
    truth = {t["case_id"]: t for t in map(json.loads, (HERE / "data/truth.jsonl").read_text().splitlines())}
    ids = sorted(i for i, t in truth.items() if t["split"] == "main")
    unv = [i for i in ids if tuple(truth[i]["stratum"]) in UNVERIFIABLE]
    out = {}
    for j in JUDGES:
        out[j] = {}
        for v in ("v0", "v1"):
            o = load(j, v)
            if not o:
                continue
            p1 = [A.q1(o.get(i, {}).get("parsed")) for i in ids]
            k = sum(1 for i in unv if A.q1(o.get(i, {}).get("parsed")) is True)
            out[j][v] = {"fooled": k, "n_unverifiable": len(unv), "fooled_share": k / len(unv), "wilson95": A.wilson(k, len(unv)),
                         "q1": A.binary(p1, [truth[i]["goal_met"] for i in ids])}
    (ROOT / "results/j1").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/j1/j1_addendum1.json").write_text(json.dumps(out, indent=2))
    for j, d in out.items():
        print(j, {v: (x["fooled"], round(x["q1"]["coverage"], 3), round(x["q1"]["accuracy_abstain_wrong"], 3)) for v, x in d.items()})


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
