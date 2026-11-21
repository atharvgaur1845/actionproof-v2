"""Build the ActionProof research site (demo/static/index.html) from demo/site_template.html and result files (CPU).

Every number on the site is read here from a file in the repository (or copied from a cited article section where
the underlying evaluation files live on the HF traces dataset); the template only lays it out.

    python scripts/build_demo_data.py   # recorded episodes -> demo/data.json
    python scripts/build_site.py        # -> demo/static/index.html
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = lambda p: (ROOT / p).read_text()


def battery() -> dict:
    """Greedy next tool per model x prefix cell (evaluation schema a4), plus the corrected readout and its flag."""
    greedy = defaultdict(Counter)
    for line in R("results/a7/battery_v2_greedy.jsonl").splitlines():
        r = json.loads(line)
        if r["schema"] == "a4":
            greedy[(r["model"], r["type"])][r["greedy_tool"]] += 1
    readout = {}
    for line in R("results/a7/battery_v2.md").splitlines():
        m = re.match(r"\| (\S+) \| ([a-e]) \| a4 \| (\d+) \| (\S+) \| [\d.]+ \| ([\d.]+) \| [\d.]+ \| \d+/50 / (\d+)/50 \|", line)
        if m:
            readout[(m.group(1), m.group(2))] = {"target": m.group(4), "p": float(m.group(5)), "agree": int(m.group(6)),
                                                 "flag": "⚠" in line.split("|")[-2] or int(m.group(6)) < 45}
    models = {"init": "A6 warm start", "a6_pcstr_d_s0": "A6 state reward (PCSTR-d)", "a6_ack_outcome_s0": "A6 acknowledgement reward",
              "a7_init": "A7 warm start", "a7_pcstr_d_noreport_s0": "A7 state reward", "a7_ack_outcome_s0": "A7 acknowledgement reward"}
    cells = {"d": "A", "c": "B", "b": "C", "e": "D"}
    out = {}
    for mk, name in models.items():
        out[name] = {cells[c]: {"greedy": dict(greedy[(mk, c)]), **readout.get((mk, c), {})} for c in cells}
    return out


def credit() -> list:
    """Recorded held-out credit episodes, three warm-start/reward combinations (scripts/build_credit_demo.py)."""
    d = json.loads(R("demo/replays/credit_templates.json"))
    items = []
    for it in d["items"]:
        lanes = {}
        for m in d["models"]:
            md = it["panels"][m]
            steps = [re.sub(r"[*`]", "", s).strip() for s in re.findall(r"^\d+\. (.+)$", md, re.M)]
            rep = re.search(r"Agent reports:\*\* `(\w+)`: _(.+?)_", md)
            db = re.search(r"\*\*Database:\*\* (.+)$", md, re.M)
            lanes[m] = {"steps": steps, "status": rep.group(1) if rep else "none", "summary": rep.group(2) if rep else "",
                        "db": re.sub(r"[*`]", "", db.group(1)) if db else ""}
        items.append({"task_id": it["task_id"], "prompt": it["prompt"], "lanes": lanes})
    return items


def theatre(data: dict) -> dict:
    """One recorded episode per trained fault type, three agents; the faulted write plus the verifier's verdict."""
    out = {}
    pick = ["Warm start (RFT v2)", "GRPO + tool-trusting reward", "GRPO + PCSTR-d (database reward)"]
    for f in ("F1_phantom", "F3_partial", "F4_timeout"):
        ep = next(e for e in data["episodes"] if e["fault"] == f)
        lanes = {}
        for n in pick:
            L = ep["lanes"][n]
            steps = [{k: s.get(k) for k in ("tool", "args", "obs", "db", "fault", "kind", "evidence", "retry")}
                     for s in L["steps"] if s["tool"] != "submit_report"]
            lanes[n] = {"steps": steps, "summary": L["summary"]}
        out[f] = {"task_id": ep["task_id"], "prompt": ep["prompt"], "lanes": lanes}
    return out


def x1() -> list:
    """Refund read-back bars from results/a8/x1_readback.md (and the addendum-2 state cell once it exists)."""
    names = {"control (task 412)": "Reply reward, all training tasks", "placebo (414q)": "Reply reward, 60 random tasks removed",
             "X1 (414p)": "Reply reward, 2 kinds of refund task removed"}
    rows = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in
            re.finditer(r"^\| (.+?) \| (\d+)/(\d+) = [\d.]+ \|$", R("results/a8/x1_readback.md"), re.M)}
    out = [[label, *rows[k]] for k, label in names.items()]
    f = ROOT / "results/a8/x1_interaction.md"
    if f.exists():
        m = re.search(r"^\| state, X1 pool \(417\) \| (\d+)/(\d+) = ", f.read_text(), re.M)
        if m:
            out.append(["Database reward, same 2 kinds removed", int(m.group(1)), int(m.group(2))])
    return out


def md_rows(path: str, ncol: int) -> list:
    """Data rows of the first markdown table with ncol columns."""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in R(path).splitlines() if l.startswith("| ")]
    return [r for r in rows if len(r) == ncol and not set(r[0]) <= set("-") and r[0] not in ("fault", "attack")]


def main() -> None:
    conf = json.loads(R("results/confirmatory.json"))
    f1 = [r for r in json.loads(R("docs/figures/figures_data.json")) if r.get("figure") == "F1" and "sampled" in r["panel"]]
    j32 = json.loads(R("results/judge/aya32b__summary.json"))["aya32b__auditkit_tc"]["ALL|FAR"]
    audit = R("results/a8/ack_class_audit.md")
    a7 = audit[audit.index("results/raw/a7/"):]
    a7 = a7[a7.index("**M4 ack_outcome, marker `reads`**"):]
    a7 = a7[:a7.index("**", 40)]  # this block only (A7 warm start, acknowledgement reward, read-back marker)
    rows = {m.group(1): (float(m.group(2)), int(m.group(3)), float(m.group(4)), int(m.group(5))) for m in
            re.finditer(r"\| (refund timeout|refund UR) \| \d+ \| [-+\d.]+ \| [-+\d.]+ \| ([\d.]+) \(n=(\d+)\) \| ([\d.]+) \(n=(\d+)\) \|", a7)}
    data = json.loads(R("demo/data.json"))
    site = {
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "verification": [{"label": r["model"], "channel": r["channel"], "rate": r["rate"], "n": r["n"], "source": r["source"]} for r in f1],
        "confirm": {"pcstr_d": conf["H1"]["means"]["pcstr_d"], "outcome": conf["H1"]["means"]["outcome"], "delta": conf["H1"]["delta"],
                    "ci": conf["H1"]["ci95"], "p": conf["H1"]["p"], "n": conf["n_tasks"], "warm": conf["per_arm_vsc"]["rft_v2"][0],
                    "tool_trust": conf["per_arm_vsc"]["tool_trust"], "vs_tt": conf["pcstr_d_vs_tool_trust"]},
        "judge": j32,
        "audit": {k: {"reader": v[0], "n_reader": v[1], "non": v[2], "n_non": v[3]} for k, v in rows.items()},
        "battery": battery(),
        "credit": credit(),
        "theatre": theatre(data),
        "x1": x1(),
        "byfault": [[r[0].replace("_", " "), r[1], *[c.split(" ")[0] for c in r[2:6]], r[6]]
                    for r in md_rows("results/a4/confirmatory_by_fault.md", 7)],
        "ledger": md_rows("results/a4/hacking_ledger.md", 7),
        "onetable": md_rows("results/one_table.md", 5)[1:] if md_rows("results/one_table.md", 5)[0][0] == "reward" else md_rows("results/one_table.md", 5),
        "agentic": [dict(zip(("model", "n", "vsc", "acc", "commit", "calls", "turns", "sec", "chars"), [c.strip() for c in l.strip("|").split("|")]))
                    for l in R("results/agentic_metrics.md").splitlines() if l.startswith("| ") and not l.startswith("| model")],
    }
    sys.path.insert(0, str(ROOT / "scripts"))
    import site_procure
    site["procure"] = site_procure.build()
    assert set(site["audit"]) == {"refund timeout", "refund UR"}, site["audit"]
    page = R("demo/site_template.html").replace("{{SITE}}", json.dumps(site, ensure_ascii=False, separators=(",", ":"))
                                                .replace("</", "<\\/")).replace("{{GIT}}", site["git"])
    out = ROOT / "demo/static/index.html"
    out.write_text(page)
    print(out, out.stat().st_size, {k: (len(v) if hasattr(v, "__len__") else v) for k, v in site.items() if k != "git"})


if __name__ == "__main__":
    main()
