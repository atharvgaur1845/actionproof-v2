"""Descriptive diagnostics of the two S4-EVAL shifts where the Stage 3 gain vanished (not registered analyses).

  s4_api  share of agent calls that use a training-time (canonical) tool name, which the shifted API rejects
          (`unknown_tool`), and share of episodes with at least one such call; per model
  s4_wf4  share of episodes whose goods-receipt step (the step the trained three-step template lacks) is absent
          from the final state; per model
  levels  mean VSC per arm on every S4 split, and mean language cost per arm (English minus language split),
          from results/s4/s4_summary.json
    python scripts/s4_diagnose.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "results/s4/eval"
TAGS = ["warm_start"] + [f"a12_rv_s{s}" for s in range(5)] + [f"s3_rv_mix_s{s}" for s in range(5)] + \
       [f"s3_matched_s{s}" for s in range(3)]
ARMS = {"warm_start": TAGS[:1], "a12_rv": TAGS[1:6], "rsc": TAGS[6:11], "control": TAGS[11:]}


def rows(tag: str, split: str):
    return [json.loads(line) for line in (E / f"{tag}__{split}.jsonl").open()]


def main() -> None:
    per: dict = {}
    for t in TAGS:
        api = rows(t, "s4_api")
        calls = [e for r in api for e in r["trace"] if e.get("by") == "agent"]
        bad = [e for e in calls if (e.get("obs") or {}).get("error") == "unknown_tool"]
        wf = rows(t, "s4_wf4")
        per[t] = {"api_canonical_call_share": len(bad) / max(len(calls), 1),
                  "api_episodes_with_canonical_call": sum(any((e.get("obs") or {}).get("error") == "unknown_tool"
                                                              for e in r["trace"] if e.get("by") == "agent")
                                                          for r in api) / len(api),
                  "wf4_receive_absent": sum(r["score"]["steps"].get("receive") == "ABSENT" for r in wf) / len(wf)}
    arms = {a: {k: sum(per[t][k] for t in ts) / len(ts) for k in per[ts[0]]} for a, ts in ARMS.items()}
    S = json.loads((ROOT / "results/s4/s4_summary.json").read_text())
    levels = {sp: {a: sum(S["splits"][sp]["models"][t]["vsc"] for t in ts) / len(ts) for a, ts in ARMS.items()}
              for sp in S["splits"]}
    lc = S["language_cost_vsc"]
    cost = {sp: {a: sum(lc[f"{t}:{sp}"]["est"] for t in ts) / len(ts) for a, ts in ARMS.items()}
            for sp in ("s4_lang_hi", "s4_lang_hinglish", "s4_lang_para")}
    (ROOT / "results/s4/s4_diagnostics.json").write_text(json.dumps(
        {"per_model": per, "per_arm": arms, "vsc_per_arm": levels, "language_cost_per_arm": cost}, indent=2))
    print(json.dumps({"vsc": {sp: {a: round(v, 3) for a, v in d.items()} for sp, d in levels.items()},
                      "cost": {sp: {a: round(v, 3) for a, v in d.items()} for sp, d in cost.items()}}, indent=0))
    for a, v in arms.items():
        print(a, {k: round(x, 3) for k, x in v.items()})


if __name__ == "__main__":
    main()
