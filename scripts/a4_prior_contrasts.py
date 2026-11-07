"""Recompute the per-seed v2 arm table and pooled contrasts cited in PREREGISTRATION_A4 §1 (read-only over results/eval)."""
import json, math, sys
from actionproof.eval.by_fault import post_ack_check
from actionproof.eval.stats import wilson, paired_bootstrap, hierarchical_bootstrap
R3="results/eval/r3"
RUNS={"init (RFT v2)":["rft_tiny_v2"],"tool_trust":["grpo_tiny_v2_tool_trust_s0","grpo_tiny_v2_tool_trust_s1"],
 "llm_judge":["grpo_tiny_v2_llm_judge_s0"],"outcome":["grpo_tiny_v2_outcome_s0","grpo_tiny_v2_outcome_s1"],
 "pcstr":["grpo_tiny_v2_pcstr_s0"],"pcstr_d":["grpo_tiny_v2_pcstr_d_s0","grpo_tiny_v2_pcstr_d_s1"],"pcstr_c":["grpo_tiny_v2_pcstr_c_s0"]}
def path(run,sp):
    if run=="rft_tiny_v2" and sp=="val": return "results/eval/r1/rft_tiny_v2__neutral__val.jsonl"
    return f"{R3}/{run}__neutral__{sp}.jsonl"
def load(run,sp): return {json.loads(l)["task_id"]:json.loads(l) for l in open(path(run,sp))}
def stats(E):
    e=list(E.values()); n=len(e); k=sum(x["vsc"] for x in e)
    g0=[x for x in e if x["verdict"]["G"]==0]; fc=sum(x["verdict"]["reported"]=="completed" for x in g0)
    acked=[x for x in e if any(s["tool"]=="issue_refund" and isinstance(s["obs"],dict) and s["obs"].get("status")=="success" for s in x["trace"])]
    pac=sum(post_ack_check(x["trace"]) for x in acked)
    anyw=[x for x in e if any(isinstance(s["obs"],dict) and s["obs"].get("status")=="success" and s["tool"] in("issue_refund","cancel_order","add_audit_note","close_ticket","restock_item","reconcile_order") for s in x["trace"])]
    paw=sum(post_ack_check(x["trace"],None) for x in anyw)
    p,lo,hi=wilson(k,n)
    return dict(n=n,vsc=round(p,3),ci=f"[{lo:.3f},{hi:.3f}]",fcr=f"{fc}/{len(g0)}={fc/max(1,len(g0)):.2f}",
      pac_refund=f"{pac}/{len(acked)}={pac/max(1,len(acked)):.2f}",pac_any=f"{paw/max(1,len(anyw)):.2f}",
      vcalls=round(sum(x["n_verify"] for x in e)/n,2),dsr=round(sum(x["verdict"]["D"]>0 for x in e)/n,3),
      ivr=round(sum(x["verdict"]["I"]>0 for x in e)/n,3),report=round(sum(x["verdict"]["reported"]!="none" for x in e)/n,3))
for sp in ["val","test_iid","test_ood_fault"]:
    print(f"\n### {sp}")
    print("| arm | run | n | VSC [Wilson95] | FCR (k/G0) | post-ack refund check | post-ack any write | verify calls/ep | DSR | IVR | report rate |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm,runs in RUNS.items():
        for r in runs:
            try: s=stats(load(r,sp))
            except FileNotFoundError: print(f"| {arm} | {r} | missing |"); continue
            print(f"| {arm} | {r} | {s['n']} | {s['vsc']} {s['ci']} | {s['fcr']} | {s['pac_refund']} | {s['pac_any']} | {s['vcalls']} | {s['dsr']} | {s['ivr']} | {s['report']} |")
def mcnemar(a,b):
    ids=set(a)&set(b); b10=sum(a[i]["vsc"]>b[i]["vsc"] for i in ids); b01=sum(a[i]["vsc"]<b[i]["vsc"] for i in ids)
    m=b10+b01; 
    if m==0: return b10,b01,1.0
    p=sum(math.comb(m,j) for j in range(0,min(b10,b01)+1))/2**m*2
    return b10,b01,min(1.0,p)
print("\n### contrasts (VSC), test_iid ∪ test_ood_fault = 400 tasks")
print("| contrast | seed pairing | Δ | 95% CI | discordant (A>B / B>A) | exact McNemar p |")
print("|---|---|---|---|---|---|")
P=lambda r:{**load(r,"test_iid"),**load(r,"test_ood_fault")}
C=[("pcstr_d","tool_trust"),("outcome","tool_trust"),("pcstr_d","outcome"),("llm_judge","tool_trust"),("pcstr","outcome"),("pcstr_c","pcstr_d"),("tool_trust","init (RFT v2)"),("outcome","init (RFT v2)")]
for a,b in C:
    A=[P(r) for r in RUNS[a]]; B=[P(r) for r in RUNS[b]]
    for i,x in enumerate(A):
        y=B[min(i,len(B)-1)]
        d,lo,hi=paired_bootstrap({k:v["vsc"] for k,v in x.items()},{k:v["vsc"] for k,v in y.items()})
        b10,b01,p=mcnemar(x,y)
        print(f"| {a} − {b} | s{i} vs s{min(i,len(B)-1)} | {d:+.4f} | [{lo:+.4f},{hi:+.4f}] | {b10}/{b01} | {p:.2g} |")
    if len(A)>1 or len(B)>1:
        bm={k:sum(y[k]["vsc"] for y in B)/len(B) for k in B[0]}
        d,lo,hi=hierarchical_bootstrap([{k:v["vsc"] for k,v in x.items()} for x in A],bm)
        print(f"| {a} − {b} | hierarchical (all seeds) | {d:+.4f} | [{lo:+.4f},{hi:+.4f}] | | |")
