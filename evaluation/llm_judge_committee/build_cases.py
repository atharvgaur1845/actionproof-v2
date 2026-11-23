"""J1 judge committee: build blinded evaluation cases from recorded ProcureSim episodes (PREREGISTRATION_J1.md §3).

Source: the registered evaluation traces of the procurement study (frozen models, already-run splits):
`results/s3/conf/*__s3_conf_full.jsonl` (English) and `results/s4/eval/*__s4_lang_{hi,hinglish}.jsonl`.

Each episode is placed in a stratum: (fault class, read-back yes/no, verified completion 0/1, report "completed" yes/no).
- Calibration set (human-labelled): up to 4 episodes from every stratum with at least 20 episodes, plus 8 extra
  Hindi/Hinglish episodes. Each episode in a stratum comes from a different model where possible.
- Main set: up to 40 further episodes per stratum, disjoint from the calibration set.

Judges see `cases.jsonl` only:
- the task request;
- the company policy and the tool catalogue (names and descriptions);
- every tool call with its arguments and returned observation;
- the final report.
Two views are produced:
- `transcript`: what a production LLM monitor would see;
- `transcript_state`: the transcript plus the final database rows for the task's purchase order.
Model identity, the injected fault and the verifier's verdict are removed. They go to `truth.jsonl`, keyed by a
random case id.

    python evaluation/llm_judge_committee/build_cases.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
SEED = 20261123
SOURCES = ("results/s3/conf/*__s3_conf_full.jsonl", "results/s4/eval/*__s4_lang_hi.jsonl",
           "results/s4/eval/*__s4_lang_hinglish.jsonl")
STATE_TABLES = ("payments", "receipts", "journal", "supplier_orders", "escalations")


def stratum(r: dict) -> tuple:
    claimed = bool(r.get("report")) and r["report"].get("status") == "completed"
    return (r["fault_class"], "read" if r["policy_reads"] > 0 else "noread", int(r["vsc"]), "claim" if claimed else "noclaim")


def tool_catalogue(system: str) -> str:
    """Names and descriptions only (the JSON schemas are long and add nothing for judging)."""
    start = system.find("[", system.find("You can call these tools:"))
    try:
        tools = json.loads(system[start:system.rindex("]") + 1])
        return "\n".join(f"- {t['function']['name']}: {t['function']['description']}" for t in tools)
    except (ValueError, KeyError):
        return ""


def policy(system: str) -> str:
    return system.split("You can call these tools:")[0].strip()


def render(r: dict, with_state: bool) -> str:
    sysmsg = r["conversation"][0]["content"]
    lines = ["## Agent instructions (policy)", policy(sysmsg), "", "## Tools available to the agent", tool_catalogue(sysmsg),
             "", "## User request", r["task_prompt"], "", "## Agent actions and tool results"]
    for i, t in enumerate(r["trace"], 1):
        if t["tool"] == "submit_report":
            continue
        lines.append(f"{i}. CALL {t['tool']}({json.dumps(t['args'], ensure_ascii=False)})")
        lines.append(f"   RESULT {json.dumps(t['obs'], ensure_ascii=False)}")
    rep = r.get("report") or {}
    lines += ["", "## Final report by the agent", json.dumps(rep, ensure_ascii=False) if rep else "(no report submitted)"]
    if with_state:
        po = r["task"].get("po_id") or _po_from_prompt(r["task_prompt"])
        lines += ["", "## Database rows for this purchase order after the episode (read-only audit snapshot)"]
        for tab in STATE_TABLES:
            rows = [x for x in r["final_state"].get(tab, []) if x.get("po_id") == po]
            lines.append(f"{tab}: {json.dumps(rows, ensure_ascii=False)}")
    return "\n".join(lines)


def _po_from_prompt(p: str) -> str | None:
    import re
    m = re.search(r"PO-[0-9A-F]{8}", p)
    return m.group(0) if m else None


def truth(r: dict) -> dict:
    s = r["score"]
    steps = s["steps"]
    committed = [v for v in steps.values() if v in ("VERIFIED_SUCCESS", "PARTIAL", "DUPLICATE")]
    occurred = "yes" if s["goal_met"] else ("partially" if committed else "no")
    return {"operation_occurred": occurred, "goal_met": bool(s["goal_met"]), "false_success": bool(s["false_success"]),
            "violation": bool(r["violation"]), "duplicate_irreversible": bool(s["duplicate_irreversible"]),
            "steps": steps, "fault_class": r["fault_class"], "read_back": r["policy_reads"] > 0,
            "escalated": bool(r["escalated"]), "honest_fail": bool(r["honest_fail"]),
            "report_status": (r.get("report") or {}).get("status")}


def main() -> None:
    rng = random.Random(SEED)
    pool: dict[tuple, list] = defaultdict(list)
    for pat in SOURCES:
        for f in sorted(glob.glob(str(ROOT / pat))):
            lang = "hi" if "lang_hi." in f else ("hinglish" if "hinglish" in f else "en")
            for line in Path(f).read_text().splitlines():
                r = json.loads(line)
                r["_lang"], r["_src"] = lang, Path(f).name
                pool[stratum(r) + (lang,)].append(r)
    by_cell: dict[tuple, list] = defaultdict(list)
    for k, v in pool.items():
        by_cell[k[:4]] += v

    def pick(cands: list, n: int, used: set) -> list:
        """n episodes from distinct models where possible, none already used."""
        rng.shuffle(cands)
        out, models = [], set()
        for r in cands:
            key = (r["_src"], r["rollout_id"])
            if key in used or (r["tag"] in models and len({c["tag"] for c in cands}) > n):
                continue
            out.append(r); models.add(r["tag"]); used.add(key)
            if len(out) == n:
                break
        return out

    used: set = set()
    calib, main = [], []
    cells = sorted(k for k, v in by_cell.items() if len(v) >= 20)
    for k in cells:
        calib += pick([r for r in by_cell[k] if r["_lang"] == "en"] or by_cell[k], 4, used)
    for lang in ("hi", "hinglish"):
        calib += pick([r for k in cells for r in pool.get(k + (lang,), [])], 4, used)
    for k in cells:
        main += pick(list(by_cell[k]), 40, used)

    OUT.mkdir(parents=True, exist_ok=True)
    cases, truths = [], []
    for split, rows in (("calibration", calib), ("main", main)):
        for r in rows:
            cid = "c" + hashlib.sha256(f"{SEED}:{r['_src']}:{r['rollout_id']}".encode()).hexdigest()[:10]
            cases.append({"case_id": cid, "split": split, "lang": r["_lang"],
                          "views": {"transcript": render(r, False), "transcript_state": render(r, True)}})
            truths.append({"case_id": cid, "split": split, "lang": r["_lang"], "stratum": list(stratum(r)),
                           "source": r["_src"], "rollout_id": r["rollout_id"], "model_tag": r["tag"], **truth(r)})
    order = list(range(len(cases)))
    rng.shuffle(order)
    (OUT / "cases.jsonl").write_text("".join(json.dumps(cases[i], ensure_ascii=False) + "\n" for i in order))
    (OUT / "truth.jsonl").write_text("".join(json.dumps(truths[i], ensure_ascii=False) + "\n" for i in order))
    print(f"strata {len(cells)}; calibration {len(calib)}; main {len(main)}")


if __name__ == "__main__":
    main()
