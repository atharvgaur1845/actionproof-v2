"""Paper-ready LaTeX tables from registered summaries only (results/p1/p1_summary.json, results/p2/p2_summary.json).

    python -m actionproof.reporting.tables       # writes paper/tables/p1.tex and paper/tables/p2.tex when the summaries exist

Each table caption names its source file, the registration, the statistic and the interval type. A table for a
study whose summary does not exist is not written; no placeholder numbers are produced.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/tables"
P1_NAMES = {"A": "RSC (reference)", "B": "A + reconciliation line", "C": "A + recovery SFT", "D": "Recovery SFT warm start + RSC recipe"}


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def sgn(x):
    return "--" if x is None else (f"+{x:.3f}" if x >= 0 else f"$-${abs(x):.3f}")


def p1(s: dict) -> str:
    t, dev = s["test"], s.get("dev", {})
    rows = []
    for arm in "ABCD":
        r = t.get(arm)
        a_mean = next(iter(t.values()))["A_mean"] if t else None
        if r is None:
            rows.append(f"{arm} & {P1_NAMES[arm]} & {f3(a_mean)} & -- & -- & -- & -- \\\\")
            continue
        seeds = ", ".join(sgn(v) for v in r["seed_deltas"])
        rows.append(f"{arm} & {P1_NAMES[arm]} & {f3(r['X_mean'])} & {sgn(r['delta'])} [{sgn(r['ci95'][0])}, {sgn(r['ci95'][1])}] & "
                    f"{r['holm_p']:.4f} & {seeds} & {'yes' if r['supported'] else 'no'} \\\\")
    winner = dev.get("winner") or "none"
    return ("\\begin{table*}[t]\\centering\\small\n\\begin{tabular}{llccccc}\\toprule\n"
            "Arm & Model & VSC (sampled) & $\\Delta$ vs A [95\\% CI] & Holm $p$ & Seed $\\Delta$s & Supported \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n"
            f"\\caption{{P1 frozen test (450 tasks, 3 seeds per arm, 4 samples at $T{{=}}1$). Paired task bootstrap on seed-averaged "
            f"task scores, one-sided, Holm over B, C, D; supported iff Holm $p<0.05$ and all three seed differences are positive "
            f"(\\texttt{{PREREGISTRATION\\_P1.md}}). Development winner: {winner}. Source: \\texttt{{results/p1/p1\\_summary.json}}.}}\n"
            "\\label{tab:p1}\\end{table*}\n")


def p2(s: dict) -> str:
    rows = []
    for fam in ("primary", "secondary"):
        for h, r in s[fam].items():
            seeds = ", ".join(sgn(v) for v in r["seed_deltas"])
            rows.append(f"{h} & {r['contrast'].replace('_', ' ')} & {f3(r['Y_mean'])} $\\to$ {f3(r['X_mean'])} & {sgn(r['delta'])} "
                        f"[{sgn(r['ci95'][0])}, {sgn(r['ci95'][1])}] & {r['holm_p']:.4f} & {seeds} & {'yes' if r['supported'] else 'no'} \\\\")
        rows.append("\\midrule")
    refs = s.get("references", {})
    ref_line = "; ".join(f"{k}: {f3(v['mean_vsc'])} ({v['decoding']}, {v['n_models']} model(s))" for k, v in refs.items())
    return ("\\begin{table*}[t]\\centering\\small\n\\begin{tabular}{llccccc}\\toprule\n"
            "Test & Contrast & VSC & $\\Delta$ [95\\% CI] & Holm $p$ & Seed $\\Delta$s & Supported \\\\\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\n"
            f"\\caption{{P2 frozen test (450 tasks, 3 seeds per arm). Primary family H1--H4 and secondary family S1--S3, Holm within "
            f"each; same rule as P1 (\\texttt{{PREREGISTRATION\\_P2.md}}). References on the same tasks: {ref_line}. "
            f"Source: \\texttt{{results/p2/p2\\_summary.json}}.}}\n\\label{{tab:p2}}\\end{{table*}}\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = []
    for name, fn, src in (("p1", p1, "results/p1/p1_summary.json"), ("p2", p2, "results/p2/p2_summary.json")):
        f = ROOT / src
        if f.exists() and json.loads(f.read_text()).get("test" if name == "p1" else "primary"):
            (OUT / f"{name}.tex").write_text(fn(json.loads(f.read_text())))
            done.append(name)
    print("tables written:", done or "none (summaries not yet available)")


if __name__ == "__main__":
    main()
