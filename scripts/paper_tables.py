"""LaTeX tables for the Stage 2 paper, generated from result files only (no number typed by hand).
    python scripts/paper_tables.py --out paper/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def j(p: str):
    return json.loads((ROOT / p).read_text())


def f2(x) -> str:
    return "--" if x is None else f"{x:.2f}"


def f3(x) -> str:
    return "--" if x is None else f"{x:.3f}"


def truth_table() -> str:
    rows = j("results/a12/reward_truth_table.json")["rows"]
    faults = [("clean", "clean"), ("phantom (success ack, nothing stored)", "phantom"),
              ("partial (success ack, half stored)", "partial"), ("lost ack (stored, timeout ack)", "lost ack"),
              ("precommit timeout (not stored, timeout ack)", "timeout before commit"),
              ("never commits (success ack every time)", "never commits")]
    behaviours = [("proceed without reading", "proceed"), ("blind retry same key on error ack", "blind retry"),
                  ("read, retry same key if missing", "read, retry"), ("read, stop and report failed if missing", "read, stop")]
    cell = {(r[0], r[1]): (r[2], r[3]) for r in rows}
    out = [r"\begin{tabular}{l" + "c" * len(behaviours) + "}", r"\toprule",
           "Fault & " + " & ".join(b for _, b in behaviours) + r" \\", r"\midrule"]
    for fk, fl in faults:
        out.append(fl + " & " + " & ".join("{}/{}".format(*cell[(fk, bk)]) for bk, _ in behaviours) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def a12_seeds() -> str:
    s = j("results/s2/a12_summary.json")
    w = j("results/eval/a12/conf_reference/warm_start__neutral__proc_conf.summary.json")
    tl = j("results/s2/a12_secondary.json")["proc_conf"]
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r" & \multicolumn{2}{c}{Phantom read-back} & \multicolumn{2}{c}{VSC} & \multicolumn{2}{c}{False success} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Seed & R$_V$ & R$_A$ & R$_V$ & R$_A$ & R$_V$ & R$_A$ \\", r"\midrule",
           rf"warm start & \multicolumn{{2}}{{c}}{{{f2(w['phantom_readback']['rate'])}}} & \multicolumn{{2}}{{c}}{{{f2(w['vsc'])}}} & \multicolumn{{2}}{{c}}{{{f2(w['false_success'])}}} \\",
           r"\midrule"]
    for seed in s["seeds"]:
        k = str(seed)
        pr = s["phantom_readback"]["per_seed"][k]
        cr = j(f"results/eval/a12/conf/a12_proc_rv_s{seed}__neutral__proc_conf.summary.json")
        ca = j(f"results/eval/a12/conf/a12_proc_ra_s{seed}__neutral__proc_conf.summary.json")
        t_rv, t_ra = tl[f"rv_s{seed}"]["task_level_phantom_readback"], tl[f"ra_s{seed}"]["task_level_phantom_readback"]
        mark = r"$^\dagger$" if seed == 0 else (r"$^\ddagger$" if seed == 2 else "")
        out.append(f"{seed}{mark} & {f2(pr['rv'])} ({t_rv['num']}/{t_rv['den']}) & {f2(pr['ra'])} ({t_ra['num']}/{t_ra['den']}) & "
                   f"{f3(cr['vsc'])} & {f3(ca['vsc'])} & {f3(cr['false_success'])} & {f3(ca['false_success'])}" + r" \\")
    v = s["vsc"]
    out += [r"\midrule",
            rf"\multicolumn{{7}}{{l}}{{VSC difference per seed: {', '.join('+' + f3(p['diff']) for p in v['per_seed'].values())}; "
            rf"exact sign-flip $p$ = 1/{round(1 / v['sign_flip_p_one_sided'])} one-sided, 2/{round(2 / v['sign_flip_p_two_sided'])} two-sided}} \\",
            r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def a14() -> str:
    d = j("results/a14/a14_summary.json")["policies"]
    r = j("results/a14/a14_summary_runtime.json")["policies"]
    order = [("warm_start", "warm start")] + [(f"a12_proc_rv_s{i}", f"R$_V$ seed {i}") for i in range(3)] + \
            [(f"a12_proc_ra_s{i}", f"R$_A$ seed {i}") for i in range(3)] + \
            [("oracle", "oracle (scripted)"), ("naive", "never-read (scripted)"), ("fixed_retry", "fixed retry (scripted)")]
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r" & \multicolumn{4}{c}{Forced read in the agent's format} & Runtime-attributed \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
           r"Policy & BCA [95\% CI] & signed $S$ & BCA timeout & Partial & BCA \\", r"\midrule"]
    for k, lab in order:
        v, w = d[k], r[k]
        ci = v["bca"]["ci95"]
        out.append(f"{lab} & {f2(v['bca']['est'])} [{f2(ci[0])}, {f2(ci[1])}] & {v['signed_S']['est']:+.2f} & "
                   f"{f2(v['bca_timeout']['est'])} & {f2(v['by_branch']['P']['correct'])} & {f2(w['bca']['est'])}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def track3() -> str:
    t = j("results/s2/track3_metrics.json")
    rows = [x for x in t["by_model_split"] if x["split"] == "proc_conf" and x["model"] in ("rv_s0", "ra_s0")]
    w = j("results/eval/a12/conf_reference/warm_start__neutral__proc_conf.summary.json")
    lat = t["latency_batch1_proc_dev"]
    out = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"Model & Verified & Claimed & Tool-call validity & Calls/ep. & Latency s (p50/p95) \\", r"\midrule",
           f"warm start & {f2(w['vsc'])} & {f2(w['vsc'] + w['false_success'])} & {f2(w['tool_call_validity'])} & "
           f"{w['mean_calls']:.1f} & {lat['warm_start']['p50']:.2f}/{lat['warm_start']['p95']:.2f}" + r" \\"]
    for x in rows:
        lab, lk = ("R$_V$ seed 0", "rv_s0") if x["model"] == "rv_s0" else ("R$_A$ seed 0", "ra_s0")
        out.append(f"{lab} & {f2(x['verified_completion'])} & {f2(x['claimed_success'])} & {f2(x['tool_call_validity'])} & "
                   f"{x['mean_calls']:.1f} & {lat[lk]['p50']:.2f}/{lat[lk]['p95']:.2f}" + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def a11() -> str:
    d = j("results/s2/a11_summary.json")
    e = d["estimands"]["ALL"]
    pols = [("warm_start", "warm start"), ("pcstr_d_s0", "PCSTR-d s0"), ("pcstr_d_s1", "PCSTR-d s1"),
            ("ack_outcome_s0", "ack s0"), ("ack_outcome_s1", "ack s1"), ("tool_trust_s0", "tool trust s0")]
    out = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Policy & Passthrough & Enforce & Enf.+Inform & $\Delta_\text{runtime}$ & $\Delta_\text{information}$ & $I$ \\",
           r"\midrule"]
    for k, lab in pols:
        dr, di = e.get(f"delta_runtime[{k}]"), e.get(f"delta_information[{k}]")
        it = d["interaction"].get(k)
        out.append(f"{lab} & {f2(e[f'vsc[{k},passthrough]'])} & {f2(e[f'vsc[{k},enforce]'])} & "
                   f"{f2(e[f'vsc[{k},enforce_inform]'])} & {dr['est']:+.2f} & {di['est']:+.2f} & "
                   + (f"{it['I']:+.3f}" if it else "--") + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/tables")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("truth", truth_table), ("a12_seeds", a12_seeds), ("a14", a14), ("track3", track3), ("a11", a11)):
        (out / f"{name}.tex").write_text("% generated by scripts/paper_tables.py; do not edit\n" + fn() + "\n")
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
