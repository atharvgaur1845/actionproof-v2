import json
import sys

d = json.load(open(sys.argv[1]))
print("| Run | train scale | read share 0-20 -> 150-200 | vote (0-20) [95%] | dQ (0-20) [95%] | dQ 150-200 | clean groups: R_V const / non-degenerate | sweep: GRPO clean c=0.05 / 1.0 | centred clean c=0.05 / 1.0 |")
print("|---|---|---|---|---|---|---|---|---|")
for k, v in d.items():
    if k.startswith("_"):
        continue
    e = v["early_effects"]
    w = v["effects_by_window"].get("150-200", {})
    cg = v["clean_group_variance_source"]
    sw = v["cost_sweep"]
    print(f"| {k} | {v['info']['train_scale']} | {v['read_share_first20']:.3f} -> {v['read_share_last50']:.3f} | "
          f"{e['vote']:+.3f} [{e['vote_ci95'][0]:+.2f}, {e['vote_ci95'][1]:+.2f}] | {e['delta_q']:+.3f} "
          f"[{e['delta_q_ci95'][0]:+.3f}, {e['delta_q_ci95'][1]:+.3f}] | {w.get('delta_q', float('nan')):+.3f} | "
          f"{cg['rv_constant']} / {cg['nonzero_var']} | {sw['all|c=0.05']['grpo_clean']:.1f} / {sw['all|c=1.0']['grpo_clean']:.1f} | "
          f"{sw['all|c=0.05']['centred_clean']:.1f} / {sw['all|c=1.0']['centred_clean']:.1f} |")
