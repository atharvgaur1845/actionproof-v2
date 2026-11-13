"""Build the demo's Leaderboard and Findings tabs from result files (no hand-typed numbers).

  demo/assets/leaderboard.md  trained arms on the untouched confirmatory splits (results/confirmatory.json) and
                              untrained Cohere models on test_iid (results/leaderboard.json), Wilson 95% CIs
  demo/assets/findings.md     docs/ABSTRACT.md without its title, plus pointers to the write-up and artifacts

Rerun after the final numbers pass:  python scripts/build_demo_assets.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path("demo/assets")
ARMS = [("rft_v2", "warm start (SFT only)"), ("tool_trust", "tool_trust: believes acks + report"),
        ("llm_judge", "LLM judge (Aya 8B)"), ("outcome", "outcome: binary DB check"), ("pcstr", "PCSTR"),
        ("pcstr_c", "PCSTR-c (verification cost)"), ("pcstr_d", "**PCSTR-d** (ours)")]


def wilson(p: float, n: int) -> tuple[float, float]:
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def leaderboard() -> str:
    conf = json.load(open("results/confirmatory.json"))
    n = conf["n_tasks"]
    lines = ["## Trained policies (Tiny Aya Global 3.35B), untouched confirmatory splits",
             "", f"`test_iid_c` ∪ `test_ood_fault_c`, n = {n} tasks, greedy. VSC = verified-success completion: the "
             "database shows the goal met **and** the report says so. One value per training seed.", "",
             "| policy | seeds | VSC (%) per seed | Wilson 95% CI (seed 0) |", "|---|---|---|---|"]
    for k, name in ARMS:
        v = conf["per_arm_vsc"][k]
        lo, hi = wilson(v[0], n)
        lines.append(f"| {name} | {len(v)} | {' / '.join(f'{x * 100:.1f}' for x in v)} | "
                     f"{lo * 100:.0f}–{hi * 100:.0f} |")
    h1 = conf["H1"]
    lines += ["", f"Registered confirmatory test H1 (PCSTR-d − outcome): **{h1['delta'] * 100:+.1f} pp** "
                  f"[{h1['ci95'][0] * 100:.1f}, {h1['ci95'][1] * 100:.1f}], p = {h1['p']:.3f}. "
                  f"PCSTR-d − tool_trust: {conf['pcstr_d_vs_tool_trust']['delta'] * 100:+.2f} pp.", "",
              "## Untrained Cohere models, `test_iid` (n = 200; neutral prompt, and the “verify every write” prompt verify_a)", "",
              "| model | VSC (%) | false-completion rate when failed (%) | verify calls / episode |",
              "|---|---|---|---|"]
    rows = [r for r in json.load(open("results/leaderboard.json")) if r["split"] == "test_iid"]
    for r in sorted(rows, key=lambda r: (-r["VSC"]["p"], r["model"], r["prompt"])):
        tag = r["model"] + ("" if r["prompt"] == "neutral" else f" + “verify” prompt ({r['prompt']})")
        lines.append(f"| {tag} | {r['VSC']['p'] * 100:.1f} [{r['VSC']['lo'] * 100:.0f}–{r['VSC']['hi'] * 100:.0f}] | "
                     f"{r['FCR']['p'] * 100:.0f} | {r['verify_calls']:.2f} |")
    lines += ["", "Sources: `results/confirmatory.json`, `results/leaderboard.json`."]
    return "\n".join(lines) + "\n"


def findings() -> str:
    body = Path("docs/ABSTRACT.md").read_text().split("\n", 1)[1].strip()
    return (body + "\n\n---\n\nFull write-up: `docs/ARTICLE_DRAFT.md`. Pre-registrations: `PREREGISTRATION*.md`. "
            "Deviation log, including our own bugs: `docs/SPEC.md` §27. Every number traces to a file in `results/`.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leaderboard.md").write_text(leaderboard())
    (OUT / "findings.md").write_text(findings())
    print("wrote", OUT / "leaderboard.md", OUT / "findings.md")


if __name__ == "__main__":
    main()
