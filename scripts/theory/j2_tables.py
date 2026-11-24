"""Markdown tables for docs/judges/* generated from results/j2/j2_summary.json (no number typed by hand).

    python scripts/theory/j2_tables.py > docs/judges/_j2_tables.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "results/j2/j2_summary.json").read_text())


def f(x, n=3):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


print("### Q1 (did the writes take effect?) against the SQLite verifier, main set (n = 660)\n")
print("| View | Judge | Coverage | Precision | Recall | F1 | False acceptance | False rejection | κ | Brier | ECE |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for view, judges in d["main_vs_verifier"].items():
    for j, m in judges.items():
        print(f"| {view} | {j} | {f(m['coverage'])} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | "
              f"{f(m['false_acceptance'])} | {f(m['false_rejection'])} | {f(m['kappa'])} | {f(m['brier'])} | {f(m['ece'])} |")
print("\n### Other questions (F1 against verifier-derived labels), main set\n")
print("| View | Judge | Q2 claimed success | Q3 false success | Q4 unrefunded duplicate | Q5 read-back (proxy) |")
print("|---|---|---|---|---|---|")
for view, judges in d["main_vs_verifier"].items():
    for j, m in judges.items():
        g = lambda k: f(m.get(k, {}).get("f1"))
        print(f"| {view} | {j} | {g('q2_claimed_success')} | {g('q3_false_success')} | {g('q4_duplicate')} | "
              f"{g('q5_vs_readback_proxy')} |")
print("\n### Fooled on unverifiable failures (transcript view, J-H1 strata)\n")
print("| Judge | Fooled / n | Share | Wilson 95% |")
print("|---|---|---|---|")
for j, m in d["fooled_unverifiable_transcript"].items():
    print(f"| {j} | {m['fooled']} / {m['n']} | {f(m['share'])} | [{f(m['wilson95'][0])}, {f(m['wilson95'][1])}] |")
print("\n### Robustness on the 198-case subset (transcript_state, Q1)\n")
print("| Judge | Order-reversal agreement | κ | Repeat agreement | κ |")
print("|---|---|---|---|---|")
for j in d["order_reversal"]:
    o, r = d["order_reversal"][j], d["repeat_run"][j]
    print(f"| {j} | {f(o['agreement_q1'])} | {f(o['kappa_q1'])} | {f(r['agreement_q1'])} | {f(r['kappa_q1'])} |")
p = d["paraphrase_aya"]
print(f"\nAya paraphrase (v2 vs v1, transcript): agreement {f(p['agreement_q1'])}, κ {f(p['kappa_q1'])}, n {p['n']}.\n")
print("### Aggregation (fit on the 76 calibration cases, evaluated on the 660 main cases)\n")
print("| View / set | Method | Coverage | Precision | Recall | F1 | FPR | κ | Brier | ECE |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k, v in d["aggregation"].items():
    for meth in ("weighted", "majority"):
        m = v[meth]
        print(f"| {k} | {meth} | {f(m['coverage'])} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | "
              f"{f(m['fpr'])} | {f(m['kappa'])} | {f(m.get('brier'))} | {f(m.get('ece'))} |")
print("\n### Inter-model agreement (five judges, Q1)\n")
print("| View | Split cases | Share | Fleiss κ |")
print("|---|---|---|---|")
for k, v in d["disagreement"].items():
    print(f"| {k} | {v['n_split_cases']} | {f(v['share'])} | {f(v['fleiss_q1'])} |")
print(f"\nHuman labels available: {d['human']['available'] or 'none'}. J2 cost (token-priced estimate): "
      f"${d['j2_cost_usd_estimate']:.3f}.")
