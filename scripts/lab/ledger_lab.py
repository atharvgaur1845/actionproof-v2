"""Write the LAB claims block for claims/claims.yaml (numbers in paper/main_v2.tex and appendix_v2.tex).

Each number is read from its result file here and checked against the displayed string before it is written, so a
rounding mismatch fails loudly instead of entering the ledger.

    python scripts/lab/ledger_lab.py > claims/claims_lab.yaml.part   # then appended by hand-check into claims.yaml
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from s2_numcheck import file_value  # noqa: E402

EA = "results/lab/exact_analysis.json"
LS = "results/lab/lab1_summary.json"
GS = "results/lab/grad_snr.json"
RR = "results/lab/train_read_rate.json"
DN = "results/lab/derived_numbers.json"
D1 = "results/ad/d1_summary.json"
DC = "results/ad2/decomposition.json"
REF = "results/lab/lab1_refs.json"


def thr(est, cond, s, q0, c):
    return [EA, ["thresholds", {"est": est, "cond": cond, "s": s, "Lf": 0.0, "q0": q0, "c": c, "rho": 0.3, "G": 8},
                 "p_star"]]


def l1(arm, f="regret"):
    return [LS, ["L1_table", arm, f]]


CLAIMS = {
    "LAB-THM1": ("Exact thresholds of the expected verify-logit update (Theorem 1 and regimes; Table thr).", "supported", {
        "g_a_s1_c01": ("0.495", *thr("grpo", "A", 1.0, 0.0, 0.01)), "g_a_s1_c05": ("0.499", *thr("grpo", "A", 1.0, 0.0, 0.05)),
        "g_a_s1_c2": ("0.500", *thr("grpo", "A", 1.0, 0.0, 0.2)), "g_a_s6_c01": ("0.654", *thr("grpo", "A", 0.6, 0.0, 0.01)),
        "g_a_s6_c05": ("0.667", *thr("grpo", "A", 0.6, 0.0, 0.05)), "g_a_s6_c2": ("0.706", *thr("grpo", "A", 0.6, 0.0, 0.2)),
        "g_q_c01": ("0.090", *thr("grpo", "A", 1.0, 0.3, 0.01)), "g_q_c05": ("0.134", *thr("grpo", "A", 1.0, 0.3, 0.05)),
        "g_q_c2": ("0.281", *thr("grpo", "A", 1.0, 0.3, 0.2)), "g_b_c01": ("0.233", *thr("grpo", "B", 1.0, 0.0, 0.01)),
        "g_b_c05": ("0.247", *thr("grpo", "B", 1.0, 0.0, 0.05)), "g_b_c2": ("0.310", *thr("grpo", "B", 1.0, 0.0, 0.2)),
        "n_c01": ("0.010", *thr("nostd", "A", 1.0, 0.0, 0.01)), "n_c05": ("0.050", *thr("nostd", "A", 1.0, 0.0, 0.05)),
        "n_c2": ("0.200", *thr("nostd", "A", 1.0, 0.0, 0.2))}),
    "LAB-TB1": ("TB1 recount from raw v2 files and exact-sign prediction.", "supported", {
        "grpo_wrong": ("0.644", EA, ["tb1_recount", "grpo", "wrong_rate"]),
        "nostd_wrong": ("0.057", EA, ["tb1_recount", "nostd", "wrong_rate"]),
        "grpoB_wrong": ("0.315", EA, ["tb1_recount", "grpo_B", "wrong_rate"])}),
    "LAB-L2": ("LAB1 mechanistic families: normaliser intervention (H4 supported) and exact gradient study (H3 not supported).",
               "supported", {
        "w_world": ("0.667", LS, ["L2N", "indep_world_std", "wrong_rate"]),
        "w_grpo": ("0.727", LS, ["L2N", "grpo", "wrong_rate"]),
        "w_task": ("0.061", LS, ["L2N", "indep_task_std", "wrong_rate"]),
        "w_nostd": ("0.045", LS, ["L2N", "nostd", "wrong_rate"]),
        "cos_grpo": ("-0.175", GS, ["per", "grpo|warm", "cosine_with_exact"]),
        "cos_min_other": ("0.989", DN, ["min_cosine_non_grpo"]),
        "snr_nostd_post": ("0.34", GS, ["per", "nostd|warm", "snr", 3]),
        "snr_fork_post": ("12.4", GS, ["per", "fork_act|warm", "snr", 3]),
        "snr_fork_order": ("0.14", GS, ["per", "fork_act|warm", "snr", 0]),
        "snr_nostd_order": ("0.12", GS, ["per", "nostd|warm", "snr", 0]),
        "snr_fork_recv": ("0.38", GS, ["per", "fork_act|warm", "snr", 1]),
        "snr_nostd_recv": ("0.35", GS, ["per", "nostd|warm", "snr", 1]),
        "snr_fork_pay": ("0.46", GS, ["per", "fork_act|warm", "snr", 2]),
        "snr_nostd_pay": ("0.33", GS, ["per", "nostd|warm", "snr", 2]),
        "crn_min": ("0.94", DN, ["crn_variance_ratio_min"]), "crn_max": ("1.12", DN, ["crn_variance_ratio_max"]),
        "epu_min": ("1.9", DN, ["episodes_per_update_ratio_min"]), "epu_max": ("5.1", DN, ["episodes_per_update_ratio_max"])}),
    "LAB-L1": ("LAB1 L1 matched-budget comparison: H1 and H2 not supported (CDVO family worse than SFB in 0/5 seeds).",
               "not_supported", {
        "grpo_e1": ("0.1834", *l1("grpo/E1")), "nostd_e1": ("0.0296", *l1("nostd/E1")), "nostd_e4": ("0.0080", *l1("nostd/E4")),
        "rloo_e1": ("0.0296", *l1("rloo/E1")), "rloo_e4": ("0.0080", *l1("rloo/E4")),
        "ppo_e1": ("0.0215", *l1("ppo/E1")), "ppo_e4": ("0.0103", *l1("ppo/E4")), "ppon_e1": ("0.0512", *l1("ppo_norm/E1")),
        "ppo_risky": ("0.785", *l1("ppo/E1", "risky_verify_rate")), "ppon_risky": ("0.531", *l1("ppo_norm/E1", "risky_verify_rate")),
        "igc_e1": ("0.0202", *l1("igc_v1/E1")), "igc_e4": ("0.0062", *l1("igc_v1/E4")),
        "fa_e1": ("0.0942", *l1("fork_act/E1")), "fa_e4": ("0.0216", *l1("fork_act/E4")),
        "fx_e1": ("0.0601", *l1("cdvo_fixed@0.25/E1")), "fx_e4": ("0.0143", *l1("cdvo_fixed@0.25/E4")),
        "ad_e1": ("0.0595", *l1("cdvo_adapt@0.25/E1")), "ad_e4": ("0.0170", *l1("cdvo_adapt@0.25/E4")),
        "ia_e1": ("0.0380", *l1("cdvo_ia@0.25/E1")), "ia_e4": ("0.0097", *l1("cdvo_ia@0.25/E4")),
        "crit_e4": ("0.085", *l1("cdvo_adapt_critic@0.25/E4")),
        "sv_nostd": ("0.039", *l1("nostd/E4", "safe_verify_rate")), "sv_rloo": ("0.042", *l1("rloo/E4", "safe_verify_rate")),
        "sv_ppo": ("0.021", *l1("ppo/E4", "safe_verify_rate")), "sv_igc": ("0.002", *l1("igc_v1/E4", "safe_verify_rate")),
        "sv_fa": ("0.033", *l1("fork_act/E4", "safe_verify_rate")), "sv_fx": ("0.114", *l1("cdvo_fixed@0.25/E4", "safe_verify_rate")),
        "sv_ad": ("0.179", *l1("cdvo_adapt@0.25/E4", "safe_verify_rate")), "sv_ia": ("0.006", *l1("cdvo_ia@0.25/E4", "safe_verify_rate")),
        "scripted": ("0.0064", REF, ["mean_regret_scripted"]),
        "l3_ppo_shift": ("0.0028", LS, ["L3G", "ppo", "test_shift"]), "l3_ppo_long": ("0.0243", LS, ["L3G", "ppo", "test_long"])}),
    "LAB-LLM": ("Language agent, costed pool, development (2 seeds per arm): training read share, greedy outcome, log decomposition.",
                "supported", {
        "g0_early": ("0.19", RR, ["ad_grpo_s0", "0-20", "share_reading"]), "g1_early": ("0.20", RR, ["ad_grpo_s1", "0-20", "share_reading"]),
        "g0_late": ("0.02", RR, ["ad_grpo_s0", "150-200", "share_reading"]), "g1_late": ("0.04", RR, ["ad_grpo_s1", "150-200", "share_reading"]),
        "n0_early": ("0.23", RR, ["ad_nostd_s0", "0-20", "share_reading"]), "n1_early": ("0.28", RR, ["ad_nostd_s1", "0-20", "share_reading"]),
        "n0_late": ("0.48", RR, ["ad_nostd_s0", "150-200", "share_reading"]), "n1_late": ("0.50", RR, ["ad_nostd_s1", "150-200", "share_reading"]),
        "jc_grpo": ("0.707", D1, ["runs", "ad_grpo_s0", "j_c"]), "jc_nostd0": ("0.740", D1, ["runs", "ad_nostd_s0", "j_c"]),
        "jc_nostd1": ("0.753", D1, ["runs", "ad_nostd_s1", "j_c"]), "reads_n1": ("1.6", D1, ["runs", "ad_nostd_s1", "reads"]),
        "reads_n0": ("1.9", D1, ["runs", "ad_nostd_s0", "reads"]),
        "jc_rsc": ("0.812", D1, ["runs", "ad_base_rsc_s0", "j_c"]), "reads_rsc": ("2.96", D1, ["runs", "ad_base_rsc_s0", "reads"]),
        "dec_clean": ("-143.6", DC, ["ad_grpo_s0", "main", "pop", "by_type", "clean"]),
        "dec_out": ("+49.6", DC, ["ad_grpo_s0", "main", "pop", "by_type", "fault_outcome"]),
        "dec_cost": ("-49.6", DC, ["ad_grpo_s0", "main", "pop", "by_type", "fault_cost"]),
        "dec_none": ("-0.36", DC, ["ad_grpo_s0", "main", "none", "net"])}),
}


def places(s: str) -> int:
    return -Decimal(s.lstrip("+")).as_tuple().exponent if "." in s else 0


def main() -> None:
    bad = []
    lines = []
    for cid, (stmt, status, nums) in CLAIMS.items():
        lines += [f"  - id: {cid}", f"    statement: \"{stmt}\"", f"    status: {status}",
                  "    prereg: PREREGISTRATION_LAB1.md" if cid != "LAB-LLM" else
                  "    prereg: docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md", "    numbers:"]
        for name, (shown, f, key) in nums.items():
            v = file_value({"file": f, "key": key})
            p = places(shown)
            if round(v, p) != round(float(shown), p):
                bad.append(f"{cid}.{name}: shown {shown} but file gives {v}")
            lines.append(f"      {name}: {{value: {shown.lstrip('+')}, file: {f}, key: {json.dumps(key)}, shown: [\"{shown}\"]}}")
        lines.append("")
    if bad:
        print("\n".join("MISMATCH " + b for b in bad), file=sys.stderr)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
