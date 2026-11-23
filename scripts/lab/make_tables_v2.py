"""Generate the appendix_v2 result tables from validated result files only.

    python scripts/lab/make_tables_v2.py      # writes paper/tables_v2/*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/tables_v2"
ARMS = [("nostd", "Dr.~GRPO"), ("rloo", "RLOO"), ("ppo", "PPO"), ("critic_v", "Value baseline"),
        ("critic_q", "Learned Q"), ("igc_v1", "igc"), ("awrbn", "AWRB-n"), ("fork_act", "Forks, full"),
        ("fork_act_nocrn", "Forks, full, no CRN"), ("cdvo_fixed@0.25", "Forks, random 25\\%"),
        ("cdvo_adapt@0.25", "Forks, adaptive 25\\%"), ("cdvo_adapt@0.1", "Forks, adaptive 10\\%"),
        ("cdvo_adapt_critic@0.25", "Adaptive + stored-mean critic"), ("cdvo_ia@0.25", "Adaptive + igc")]


def f4(x):
    return "--" if x is None else f"{x:.4f}"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = json.loads((ROOT / "results/lab/lab1_summary.json").read_text())
    T = d["L1_table"]
    # per-seed regret, E4
    rows = []
    for k, name in ARMS:
        t = T.get(f"{k}/E4")
        if not t:
            continue
        seeds = [t["seed"][str(s)]["regret"] for s in range(5)]
        rows.append(f"{name} & " + " & ".join(f4(x) for x in seeds) + f" & {f4(t['regret'])} \\\\")
    (OUT / "l1_perseed_e4.tex").write_text(
        "\\begin{tabular}{lcccccc}\n\\toprule\nEstimator & s0 & s1 & s2 & s3 & s4 & mean\\\\\n\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # gates
    g = d["gates"]
    gl = []
    for h in ("H1|E1", "H1|E4", "H2|E1", "H2|E4"):
        x = g[h]
        gl.append(f"{h.replace('|', ' ')} & {x['sfb'].replace('_', '\\_')} & {f4(x['cand_regret'])} & {f4(x['sfb_regret'])} & "
                  f"{x['rel_improvement']:+.2f} & {x['seeds_better']}/5 & {x['p_holm']:.2f} \\\\")
    (OUT / "l1_gates.tex").write_text(
        "\\begin{tabular}{lcccccc}\n\\toprule\nTest & SFB & cand. & SFB & rel. & seeds & Holm $p$\\\\\n\\midrule\n" +
        "\n".join(gl) + "\n\\bottomrule\n\\end{tabular}\n")
    # L2N
    n = d["L2N"]
    nl = [f"{a.replace('_', '\\_')} & {n[a]['wrong_rate']:.3f} & {n[a]['mean_regret']:.4f} \\\\"
          for a in ("grpo", "indep_world_std", "indep_task_std", "nostd")]
    (OUT / "l2n.tex").write_text("\\begin{tabular}{lcc}\n\\toprule\nScale & wrong-decision rate & regret\\\\\n\\midrule\n" +
                                 "\n".join(nl) + "\n\\bottomrule\n\\end{tabular}\n")
    # grad SNR
    gs = json.loads((ROOT / "results/lab/grad_snr.json").read_text())
    gl = []
    for e, name in (("grpo", "GRPO"), ("nostd", "Dr.~GRPO"), ("igc_v1", "igc"), ("fork_act", "Forks"),
                    ("fork_act_nocrn", "Forks, no CRN")):
        v = gs["per"][f"{e}|warm"]
        gl.append(f"{name} & {v['cosine_with_exact']:+.3f} & " + " & ".join(f"{x:.2f}" for x in v["snr"]) + " \\\\")
    (OUT / "grad.tex").write_text(
        "\\begin{tabular}{lccccc}\n\\toprule\nEstimator & cos & order & receive & pay & never-faulted\\\\\n\\midrule\n" +
        "\n".join(gl) + "\n\\bottomrule\n\\end{tabular}\n")
    # L3G
    l3 = d["L3G"]
    ll = []
    for k, name in (("nostd", "Dr.~GRPO"), ("rloo", "RLOO"), ("ppo", "PPO"), ("critic_q", "Learned Q"),
                    ("igc_v1", "igc"), ("fork_act", "Forks, full"), ("cdvo_adapt@0.25", "Forks, adaptive"),
                    ("cdvo_ia@0.25", "Adaptive + igc")):
        v = l3[k]
        ll.append(f"{name} & " + " & ".join(f4(v[s]) for s in ("train", "test_iid", "test_shift", "test_long")) + " \\\\")
    (OUT / "l3g.tex").write_text(
        "\\begin{tabular}{lcccc}\n\\toprule\nEstimator & train & iid & shift & long\\\\\n\\midrule\n" + "\n".join(ll) +
        "\n\\bottomrule\n\\end{tabular}\n")
    # LLM training read rates
    rr = json.loads((ROOT / "results/lab/train_read_rate.json").read_text())
    rl = []
    for t in ("ad_grpo_s0", "ad_grpo_s1", "ad_nostd_s0", "ad_nostd_s1", "ad_awrb_s0", "ad_awrbn_s0"):
        v = rr[t]
        rl.append(f"{t.replace('_', '\\_')} & " + " & ".join(f"{v[w]['share_reading']:.2f}" for w in
                                                          ("0-20", "20-50", "50-100", "100-150", "150-200")) +
                  f" & {v['150-200']['zero_var_groups']:.2f} \\\\")
    (OUT / "read_rate.tex").write_text(
        "\\begin{tabular}{lcccccc}\n\\toprule\nRun & 0--20 & 20--50 & 50--100 & 100--150 & 150--200 & zero-var\\\\\n"
        "\\midrule\n" + "\n".join(rl) + "\n\\bottomrule\n\\end{tabular}\n")
    print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
