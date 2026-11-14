"""Build the static demo page (demo/static/index.html) from demo/data.json and the SVG charts (CPU).

    python scripts/build_demo_data.py && python scripts/build_static_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
import charts as C  # noqa: E402


def main() -> None:
    d = json.loads((ROOT / "demo/data.json").read_text())
    ch, models = d["charts"], d["meta"]["models"]
    keep = ("tool", "args", "ack", "db", "fault", "kind", "evidence", "retry", "plan", "harness")
    eps = [{"task_id": x["task_id"], "fault": x["fault"], "prompt": x["prompt"],
            "lanes": {n: {"summary": L["summary"], "steps": [{k: s[k] for k in keep if k in s} for s in L["steps"]]}
                      for n, L in x["lanes"].items()}} for x in d["episodes"]]
    page = (ROOT / "demo/static_template.html").read_text()
    grey = lambda rows, pick=lambda r: False: [dict(r, color="var(--pen)" if pick(r) else "var(--n-400)") for r in rows]
    parts = {
        "C1": C.c1_vsc(ch["c1"], ch["c1_contrasts"]),
        "C6": C.c6_faults(ch["c6"], models),
        "C2": C.c2_channel(ch["c2"]),
        "C3": C.c3_x1(ch["c3"], ch["c3_diff"], ch["c3_step0"]),
        "C4": C.bars(grey(ch["c4"], lambda r: "AuditKIT" in r["judge"] and "32B" in r["judge"]), "far", "judge",
                     "Judges accept false completions",
                     "Share of 476 claimed-completed episodes, failed per the database, that each judge accepted. Thin line: "
                     "the subset where the failure is visible in the transcript. Whiskers: 95% CI.", color_key="_", ci_key="ci95",
                     second_key="far_visible"),
        "C5": C.c5_agentic(ch["c5"]),
        "GIT": d["meta"]["git"],
        "DATA": json.dumps({"models": models, "episodes": eps}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    }
    for k, v in parts.items():
        page = page.replace("{{" + k + "}}", v)
    out = ROOT / "demo/static/index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
