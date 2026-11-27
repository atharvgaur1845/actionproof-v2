"""Interactive impact scenario calculator: demo/impact/index.html (self-contained, offline, no external dependency).

Every rate on the page is MEASURED and read here from result files (txn_test summaries; numerators and denominators
kept). Every dollar figure is a PROJECTION from user-adjustable assumptions whose defaults are docs/upgrade/ROI.md §3
(the IMPACT1 block of claims/claims.yaml). The cost model is scripts/impact_txn.py's, extended by three optional terms
that default to off (flagged-exception cost = exception cost, partial-loss cost = 0, compute and latency shown apart
from the total). With the defaults the page reproduces results/impact/impact_txn.json; the build fails if it does not.

    python scripts/build_impact_page.py          # writes demo/impact/index.html

Add an arm (for example the Recovery Gate) with one line in AGENTS. Tag syntax: "[dir:]name[@split]" reads
results/<dir or txn>/<name>__<split or txn_test>.summary.json. An arm whose files are missing is skipped with a warning.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo/impact/index.html"
IMPACT = ROOT / "results/impact/impact_txn.json"
SPLIT = "txn_test"

# (label, tags, family). Families: scripted | reference | learned | sel1 | gate. Order is fixed: colour follows the arm.
AGENTS: list[tuple[str, list[str], str]] = [
    ("B0 trust the acknowledgement (scripted)", ["B0"], "scripted"),
    ("B1 final-state check only (scripted)", ["B1"], "scripted"),
    ("B3 bounded retry + idempotency key (scripted)", ["B3"], "scripted"),
    ("Base Tiny Aya Global, prompted", ["tag_base"], "learned"),
    ("W plain SFT warm start", ["W"], "learned"),
    ("Dw_aug randomised recovery SFT (3 seeds)", ["Dwaug_s0", "Dwaug_s1", "Dwaug_s2"], "learned"),
    ("D recovery SFT + GRPO (3 seeds)", ["D_s0", "D_s1", "D_s2"], "learned"),
    ("SEL1 CTL (3 seeds)", ["sel1:CTL_s0", "sel1:CTL_s1", "sel1:CTL_s2"], "sel1"),
    ("SEL1 SEL (3 seeds)", ["sel1:SEL_s0", "sel1:SEL_s1", "sel1:SEL_s2"], "sel1"),
    ("B4 transaction-aware reconciliation (scripted reference)", ["B4"], "reference"),
    # ("D + Recovery Gate (3 seeds)", ["recovery_gate:RG_D_s0", "recovery_gate:RG_D_s1", "recovery_gate:RG_D_s2"], "gate"),
]
FAMILIES = {"scripted": "Scripted baseline", "reference": "Scripted reference (knows the fault taxonomy)",
            "learned": "Model (Tiny Aya Global, 3.35B)", "sel1": "Model, SEL1 continuation", "gate": "Model + Recovery Gate"}

# docs/upgrade/ROI.md §3 (sources and their quality are stated there); = scripts/impact_txn.py SCEN
SCENARIOS = {
    "conservative": {"f": 0.005, "value": 2000, "exception": 25, "mult": 2, "recovery": 0.70, "gpu": 2.50},
    "moderate": {"f": 0.02, "value": 5000, "exception": 60, "mult": 3, "recovery": 0.60, "gpu": 5.00},
    "optimistic": {"f": 0.05, "value": 10000, "exception": 175, "mult": 5, "recovery": 0.50, "gpu": 5.00},
}
CALLS_PER_GPU_HOUR = 72000  # ROI.md §3: measured, 9,600 tickets per H200-hour x 7.5 calls (results/roi/throughput.json)
DEFAULT_SECONDS_PER_CALL = 2.0  # assumption, no source: user-adjustable
DEFAULT_VOLUME = 100000  # workflows per year: illustrative, no source: user-adjustable
METRICS = ("VTC", "FCR", "DMR_all", "FCompR_all", "URR_all", "CER", "unsafe_unresolvable", "PRS", "coverage", "STR")


def resolve(tag: str) -> tuple[Path, str, str]:
    d, name = tag.split(":", 1) if ":" in tag else ("txn", tag)
    name, split = name.split("@", 1) if "@" in name else (name, SPLIT)
    return ROOT / f"results/{d}/{name}__{split}.summary.json", name, split


def load_tag(tag: str) -> dict | None:
    path, name, split = resolve(tag)
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    a = s["txn"]["all"]
    m = {k: {"num": a[k]["num"], "den": a[k]["den"]} for k in METRICS}
    n = a["n"]
    rates = {k: (v["num"] / v["den"] if v["den"] else 0.0) for k, v in m.items()}
    prs = m["PRS"]
    rates["partial_unrepaired"] = (prs["den"] - prs["num"]) / n  # partial-fault episodes not repaired, per episode
    rates["escalated"] = 1 - rates["coverage"]
    per_task = 1 + int(s["samples"]) if "samples" in s else 1
    return {"tag": tag, "name": name, "split": split, "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": n, "episodes_per_task": per_task,
            "kind": "scripted policy" if "policy" in s else "model", "counts": m, "rates": rates,
            "calls": a["cost"]["calls"], "model": s.get("model"), "adapter": s.get("adapter")}


def cost(r: dict, a: dict) -> float:
    """scripts/impact_txn.py::cost plus the optional flagged/partial terms (defaults reproduce it exactly)."""
    e, m = a["exception"], a["mult"]
    other = max(0.0, 1 - r["VTC"] - r["FCR"])
    return 1000 * (r["FCR"] * e * m + r["DMR_all"] * (a["value"] * (1 - a["recovery"]) + e * m)
                   + r["FCompR_all"] * e * m + other * a.get("flagged", e)
                   + r["partial_unrepaired"] * a.get("partial", 0.0))


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build() -> dict:
    agents = []
    for label, tags, fam in AGENTS:
        rows = [load_tag(t) for t in tags]
        if any(r is None for r in rows):
            missing = [t for t, r in zip(tags, rows) if r is None]
            print(f"skip {label!r}: missing {[str(resolve(t)[0].relative_to(ROOT)) for t in missing]}", file=sys.stderr)
            continue
        splits = {r["split"] for r in rows}
        pooled = {k: {"num": sum(r["counts"][k]["num"] for r in rows), "den": sum(r["counts"][k]["den"] for r in rows)}
                  for k in METRICS}
        n = sum(r["n"] for r in rows)
        ept = rows[0]["episodes_per_task"]
        if rows[0]["kind"] == "scripted policy":
            scope = f"deterministic scripted policy; 1 episode per task; {n} episodes"
        else:
            scope = (f"{len(rows)} seed{'s' if len(rows) > 1 else ''} x {rows[0]['n'] // ept} tasks x {ept} episodes "
                     f"(1 greedy + {ept - 1} sampled) = {n} episodes")
        sens = {}
        for sname, sc in SCENARIOS.items():
            cs = [cost(r["rates"], sc) for r in rows]
            sens[sname] = {"faulted_mean": sum(cs) / len(cs), "faulted_min": min(cs), "faulted_max": max(cs),
                           "workflows": sc["f"] * sum(cs) / len(cs)}
        agents.append({"label": label, "family": fam, "family_label": FAMILIES[fam], "tags": tags,
                       "split": "/".join(sorted(splits)), "scope": scope, "n": n, "pooled": pooled,
                       "per_tag": [{"tag": r["tag"], "path": r["path"], "sha256": r["sha256"], "n": r["n"],
                                    "rates": r["rates"], "calls": r["calls"]} for r in rows],
                       "calls_mean": sum(r["calls"] for r in rows) / len(rows), "sensitivity": sens})
    check = verify(agents)
    return {"agents": agents, "scenarios": SCENARIOS, "calls_per_gpu_hour": CALLS_PER_GPU_HOUR,
            "defaults": {**SCENARIOS["moderate"], "flagged": SCENARIOS["moderate"]["exception"], "partial": 0.0,
                         "seconds_per_call": DEFAULT_SECONDS_PER_CALL, "volume": DEFAULT_VOLUME},
            "check": check, "built": {"commit": git_head(),
                                      "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                      "impact_json_sha256": hashlib.sha256(IMPACT.read_bytes()).hexdigest()}}


def verify(agents: list[dict]) -> dict:
    """Every arm that results/impact/impact_txn.json also prices must match it in every scenario (both units)."""
    ref = json.loads(IMPACT.read_text())["agents"]
    by_tags = {tuple(v["tags"]): (k, v) for k, v in ref.items()}
    matched = []
    for a in agents:
        hit = by_tags.get(tuple(a["tags"]))
        if hit is None:
            continue
        label, v = hit
        for s in SCENARIOS:
            mine, theirs = a["sensitivity"][s], v["projected_cost_per_1000_faulted"][s]
            for k_m, k_t in (("faulted_mean", "mean"), ("faulted_min", "min"), ("faulted_max", "max")):
                if abs(mine[k_m] - theirs[k_t]) > 1e-6 * max(1.0, abs(theirs[k_t])):
                    raise SystemExit(f"impact mismatch {a['label']} {s} {k_m}: {mine[k_m]} vs {theirs[k_t]}")
            if abs(mine["workflows"] - v["per_1000_workflows"][s]) > 1e-6 * max(1.0, v["per_1000_workflows"][s]):
                raise SystemExit(f"impact mismatch {a['label']} {s} per_1000_workflows")
        matched.append(label)
    return {"reproduces": matched, "of": len(ref)}


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ActionProof: what do wrong writes cost? (scenario calculator)</title>
<style>
:root{color-scheme:light;--bg:#ffffff;--panel:#f6f6f4;--line:#dddddb;--text:#191919;--muted:#5c5c5c;
 --accent:#1b64be;--warn-bg:#fff4d6;--warn-line:#d28e05;--focus:#1757ac;
 --f-scripted:#9a9a9a;--f-reference:#3d3d3d;--f-learned:#257de0;--f-sel1:#00b894;--f-gate:#f47230}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#191919;--panel:#222222;--line:#3a3a3a;--text:#efefef;
 --muted:#a8a8a8;--accent:#6aabf1;--warn-bg:#3a2e10;--warn-line:#d28e05;--focus:#84b9f3;
 --f-scripted:#8a8a8a;--f-reference:#d6d6d6;--f-learned:#338ced;--f-sel1:#00a888;--f-gate:#e76624}}
*{box-sizing:border-box}
html{font-size:16px}
body{margin:0;background:var(--bg);color:var(--text);font:400 15px/1.5 system-ui,-apple-system,"Segoe UI",Inter,"Noto Sans",Roboto,Arial,sans-serif}
.banner{position:sticky;top:0;z-index:5;background:var(--warn-bg);border-bottom:2px solid var(--warn-line);
 padding:10px 20px;font-weight:600;font-size:14px}
.banner span{font-weight:400;color:var(--text)}
main{max-width:1240px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:28px;line-height:1.2;margin:0 0 8px;letter-spacing:-0.01em}
h2{font-size:19px;margin:40px 0 10px}
p{margin:0 0 10px;max-width:78ch}
.lede{color:var(--muted);font-size:16px}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--panel);padding:1px 4px;border-radius:3px}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 6px}
button{font:inherit;font-size:14px;padding:6px 14px;border:1px solid var(--line);background:var(--bg);color:var(--text);
 border-radius:6px;cursor:pointer}
button[aria-pressed="true"]{border-color:var(--accent);color:var(--accent);font-weight:600}
button:focus-visible,input:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px 18px;margin-top:10px}
.field{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.field label{display:block;font-weight:600;font-size:14px}
.field .row{display:flex;align-items:center;gap:8px;margin:6px 0 4px}
.field input{width:9.5em;font:inherit;font-size:15px;padding:4px 8px;border:1px solid var(--line);border-radius:5px;
 background:var(--bg);color:var(--text)}
.field .unit{color:var(--muted);font-size:13px}
.field .src{color:var(--muted);font-size:12.5px;line-height:1.35}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;white-space:nowrap}
th{background:var(--panel);font-weight:600;font-size:12.5px;color:var(--muted);position:sticky;top:0}
td.l,th.l{text-align:left;white-space:normal;min-width:17em}
tr:last-child td{border-bottom:0}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:baseline}
.fam{display:block;color:var(--muted);font-size:12px}
.src-list{margin-top:4px;font-size:11.5px;overflow-wrap:anywhere}
.src-list code{font-size:11px;padding:0 2px}
.bar{height:8px;border-radius:0 4px 4px 0;min-width:1px;margin-top:5px}
.big{font-size:15px;font-weight:600}
.muted{color:var(--muted)}
.small{font-size:12.5px}
.note{border-left:3px solid var(--line);padding:2px 12px;color:var(--muted);margin:10px 0}
footer{margin-top:44px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:12px}
@media (max-width:700px){h1{font-size:23px}main{padding:20px 14px 48px}.grid{grid-template-columns:1fr}
 .banner{font-size:13px;padding:8px 14px}.banner span{display:none}}
</style>
</head>
<body>
<div class="banner" role="note">Scenario model from measured simulator rates; not validated real-world ROI.
<span>Rates are measured on ProcureSim's locked txn_test; every dollar figure depends on the assumptions below.</span></div>
<main>
<h1>What does a wrong write cost?</h1>
<p class="lede">A tool can reply <code>ok</code> when nothing was stored, or time out after the write committed. Each agent
below was measured on the same 440 locked tasks. Change the assumptions to price its mistakes for your own workflow.</p>

<h2 id="assumptions">Assumptions</h2>
<p class="small muted">Presets are the three brackets of <code>docs/upgrade/ROI.md</code> §3. The defaults (moderate) reproduce
<code>results/impact/impact_txn.json</code> exactly.</p>
<div class="presets" role="group" aria-label="Scenario presets">
 <button type="button" data-preset="conservative">Conservative</button>
 <button type="button" data-preset="moderate" aria-pressed="true">Moderate</button>
 <button type="button" data-preset="optimistic">Optimistic</button>
</div>
<div class="grid" id="fields"></div>

<h2 id="results">Projected cost of each agent's mistakes</h2>
<p class="small muted">Two units, as in write-up §24: per 1,000 workflows that meet an ambiguous-execution fault (the
benchmark's mix), and per 1,000 workflows in all (the first, times the fault incidence). Clean workflows are assumed
handled correctly by every agent, which favours the acknowledgement-trusting agent. Hover a rate for its count.</p>
<div class="tablewrap"><table id="results-table" aria-live="polite"></table></div>
<p class="note small">Compute and agent time are shown apart from the total: they are small next to the error costs and
use separate assumptions (GPU price, seconds per call).</p>

<h2 id="rates">Measured rates and their sources</h2>
<p class="small muted">Counts are pooled over seeds. Model arms run each task once greedily and twice sampled. Scripted
policies run each task once. Every count is in the named summary file, which is in the repository and on the
Hugging Face traces dataset.</p>
<div class="tablewrap"><table id="rates-table"></table></div>

<h2 id="sensitivity">Sensitivity: low, base and high brackets</h2>
<p class="small muted">Each column uses one bracket of ROI.md §3 in full (fault incidence, value, exception cost,
late-discovery multiplier, recovery rate). The range over seeds is in brackets.</p>
<div class="tablewrap"><table id="sens-table"></table></div>

<h2 id="model">How the cost is computed</h2>
<p>Per 1,000 faulted workflows, for each episode outcome measured on the simulator:</p>
<ul>
<li><b>False completion</b> (reported done, not done): exception cost x late-discovery multiplier.</li>
<li><b>Duplicate mutation</b>: value x (1 - recovery rate) + exception cost x multiplier.</li>
<li><b>False refund</b> (a committed payment refunded): exception cost x multiplier.</li>
<li><b>Every other episode that is not a verified completion</b> (failed report, escalation, stall): the flagged-exception
cost, because a person sees it at once.</li>
<li><b>Optional:</b> an extra loss per partial transaction left unrepaired (default 0; the IMPACT1 model charges it only
as an exception).</li>
</ul>
<p>Per 1,000 workflows in all = the above x fault incidence. Per year = that x volume / 1,000. Seed arms: the mean over
seeds, with the range. This is <code>scripts/impact_txn.py</code>'s model; the build checks the defaults against its output.</p>
<p class="note">What this page cannot tell you: the real fault incidence of your systems, how an agent behaves outside a
simulator, or the cost of building the scripted reconciliation checks (B4), which need the fault taxonomy written down
in advance. B4 is a reference, not a trained agent.</p>
<footer id="foot"></footer>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
(function(){
"use strict";
var D=JSON.parse(document.getElementById("data").textContent);
var FIELDS=[
 {k:"f",label:"Fault incidence per workflow",unit:"%",pct:true,step:0.1,min:0,max:100,
  src:"Assumption, bracketed by APQC's 0.8-2.0% duplicate-or-erroneous disbursement rate (ROI.md §3)."},
 {k:"value",label:"Transaction value",unit:"$",step:100,min:0,src:"Illustrative; no source."},
 {k:"exception",label:"Cost to resolve one exception by hand",unit:"$",step:5,min:0,
  src:"Weak: AP per-invoice cost benchmarks and one vendor estimate (ROI.md §3)."},
 {k:"mult",label:"Late-discovery multiplier",unit:"x",step:0.5,min:1,src:"Assumption: an error found late costs this many exceptions."},
 {k:"recovery",label:"Duplicate payment recovery rate",unit:"%",pct:true,step:5,min:0,max:100,
  src:"Weak: an unsourced 60-70% blog estimate (ROI.md §3)."},
 {k:"flagged",label:"Cost of a flagged exception (failed report, escalation, stall)",unit:"$",step:5,min:0,
  src:"Defaults to the exception cost, as in IMPACT1."},
 {k:"partial",label:"Extra loss per unrepaired partial transaction",unit:"$",step:10,min:0,
  src:"Default 0 (IMPACT1 charges it only as an exception)."},
 {k:"volume",label:"Workflows per year",unit:"",step:1000,min:0,src:"Illustrative; scales the yearly column only."},
 {k:"gpu",label:"GPU price",unit:"$/hour",step:0.5,min:0,src:"HF Jobs H200 listing (ROI.md §3)."},
 {k:"seconds_per_call",label:"Seconds per tool call",unit:"s",step:0.5,min:0,
  src:"Assumption, no source: tool latency plus generation."}
];
var A=Object.assign({},D.defaults);
var fmt0=new Intl.NumberFormat("en-US",{maximumFractionDigits:0});
var fmt2=new Intl.NumberFormat("en-US",{minimumFractionDigits:2,maximumFractionDigits:2});
function money(x){return "$"+fmt0.format(Math.round(x));}
function rate(x){return (x).toFixed(3);}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c];});}
function cost(r,a){
 var e=a.exception,m=a.mult,other=Math.max(0,1-r.VTC-r.FCR);
 return 1000*(r.FCR*e*m+r.DMR_all*(a.value*(1-a.recovery)+e*m)+r.FCompR_all*e*m+other*a.flagged+r.partial_unrepaired*a.partial);
}
function sw(f){return '<span class="sw" style="background:var(--f-'+f+')" aria-hidden="true"></span>';}
function buildFields(){
 var g=document.getElementById("fields");
 g.innerHTML=FIELDS.map(function(f){
  var v=f.pct?A[f.k]*100:A[f.k];
  return '<div class="field"><label for="in-'+f.k+'">'+esc(f.label)+'</label><div class="row">'+
   '<input id="in-'+f.k+'" type="number" inputmode="decimal" step="'+f.step+'" min="'+f.min+'"'+(f.max!=null?' max="'+f.max+'"':'')+
   ' value="'+(+v.toFixed(4))+'"><span class="unit">'+esc(f.unit)+'</span></div><div class="src">'+esc(f.src)+'</div></div>';
 }).join("");
 FIELDS.forEach(function(f){
  document.getElementById("in-"+f.k).addEventListener("input",function(ev){
   var x=parseFloat(ev.target.value);if(!isFinite(x))return;
   A[f.k]=f.pct?x/100:x;
   if(f.k==="exception"&&!flaggedEdited){A.flagged=x;document.getElementById("in-flagged").value=x;}
   if(f.k==="flagged")flaggedEdited=true;
   setPressed(null);render();
  });
 });
}
var flaggedEdited=false;
function setPressed(name){
 document.querySelectorAll("[data-preset]").forEach(function(b){b.setAttribute("aria-pressed",String(b.dataset.preset===name));});
}
document.querySelectorAll("[data-preset]").forEach(function(b){
 b.addEventListener("click",function(){
  var s=D.scenarios[b.dataset.preset];
  ["f","value","exception","mult","recovery","gpu"].forEach(function(k){A[k]=s[k];});
  A.flagged=s.exception;A.partial=0;flaggedEdited=false;
  buildFields();setPressed(b.dataset.preset);render();
 });
});
function render(){
 var rows=D.agents.map(function(ag){
  var cs=ag.per_tag.map(function(t){return cost(t.rates,A);});
  var mean=cs.reduce(function(s,x){return s+x;},0)/cs.length;
  return {ag:ag,mean:mean,min:Math.min.apply(null,cs),max:Math.max.apply(null,cs),wf:A.f*mean};
 });
 var top=Math.max.apply(null,rows.map(function(r){return r.wf;}))||1;
 var h='<thead><tr><th class="l" scope="col">Agent</th><th scope="col">Verified<br>completion</th><th scope="col">False<br>completion</th>'+
  '<th scope="col">Duplicate<br>mutation</th><th scope="col">False<br>refund</th><th scope="col">Flagged<br>(not verified)</th>'+
  '<th scope="col">Per 1,000<br>faulted workflows</th><th scope="col">Per 1,000<br>workflows</th><th scope="col">Per year</th>'+
  '<th scope="col">Compute per 1,000<br>workflows; agent time</th></tr></thead><tbody>';
 rows.forEach(function(r){
  var ag=r.ag,p=ag.pooled,n=ag.n;
  function cell(k){var c=p[k];return '<td title="'+c.num+' of '+c.den+' episodes">'+rate(c.num/c.den)+'</td>';}
  var flagged=n-p.VTC.num-p.FCR.num;
  var range=ag.per_tag.length>1?'<div class="small muted">'+money(r.min)+' to '+money(r.max)+'</div>':'';
  var compute=ag.calls_mean*1000/D.calls_per_gpu_hour*A.gpu;
  h+='<tr><td class="l">'+sw(ag.family)+esc(ag.label)+'<span class="fam">'+esc(ag.family_label)+'</span></td>'+
   cell("VTC")+cell("FCR")+cell("DMR_all")+cell("FCompR_all")+
   '<td title="'+Math.max(0,flagged)+' of '+n+' episodes">'+rate(Math.max(0,flagged)/n)+'</td>'+
   '<td><span class="big">'+money(r.mean)+'</span>'+range+'</td>'+
   '<td><span class="big">'+money(r.wf)+'</span><div class="bar" style="width:'+Math.max(1,Math.round(110*r.wf/top))+'px;background:var(--f-'+ag.family+')"></div></td>'+
   '<td>'+money(r.wf*A.volume/1000)+'</td>'+
   '<td>$'+fmt2.format(compute)+'<div class="small muted">'+(ag.calls_mean*A.seconds_per_call).toFixed(1)+' s; '+ag.calls_mean.toFixed(2)+' calls</div></td></tr>';
 });
 document.getElementById("results-table").innerHTML=h+"</tbody>";
}
function ratesTable(){
 var K=[["VTC","Verified completion"],["FCR","False completion"],["DMR_all","Duplicate mutation"],["FCompR_all","False refund"],
  ["URR_all","Unsafe re-execution"],["CER","Escalation when it is the safe action"],["unsafe_unresolvable","Unsafe when evidence is unresolvable"],
  ["PRS","Partial transaction repaired"]];
 var h='<thead><tr><th class="l" scope="col">Agent, split, seed scope and source (sha256, first 12)</th>'+K.map(function(k){return '<th scope="col">'+k[1].replace(/ (?=\S+$)/,"<br>")+'</th>';}).join("")+
  '</tr></thead><tbody>';
 D.agents.forEach(function(ag){
  h+='<tr><td class="l">'+sw(ag.family)+esc(ag.label)+'<span class="fam">'+esc(ag.split)+'; '+esc(ag.scope)+'</span>'+
   '<span class="fam src-list">'+ag.per_tag.map(function(t){return '<code>'+esc(t.path)+'</code> <span style="white-space:nowrap">sha256 '+t.sha256.slice(0,12)+'</span>';}).join("<br>")+'</span></td>'+
   K.map(function(k){var c=ag.pooled[k[0]];return '<td>'+c.num+' / '+c.den+'<div class="small muted">'+(c.den?rate(c.num/c.den):"n/a")+'</div></td>';}).join("")+'</tr>';
 });
 document.getElementById("rates-table").innerHTML=h+"</tbody>";
}
function sensTable(){
 var S=["conservative","moderate","optimistic"];
 var h='<thead><tr><th class="l" scope="col">Agent</th>'+S.map(function(s){return '<th scope="col">Per 1,000 faulted<br>'+s+'</th>';}).join("")+
  S.map(function(s){return '<th scope="col">Per 1,000 workflows<br>'+s+'</th>';}).join("")+'</tr></thead><tbody>';
 D.agents.forEach(function(ag){
  h+='<tr><td class="l">'+sw(ag.family)+esc(ag.label)+'</td>'+S.map(function(s){var x=ag.sensitivity[s];
   return '<td>'+money(x.faulted_mean)+(ag.per_tag.length>1?'<div class="small muted">['+money(x.faulted_min)+', '+money(x.faulted_max)+']</div>':'')+'</td>';}).join("")+
   S.map(function(s){return '<td>'+money(ag.sensitivity[s].workflows)+'</td>';}).join("")+'</tr>';
 });
 document.getElementById("sens-table").innerHTML=h+"</tbody>";
}
function foot(){
 var b=D.built;
 document.getElementById("foot").innerHTML='Built by <code>scripts/build_impact_page.py</code> at commit '+esc(b.commit)+', '+esc(b.utc)+
  '. Build check: with the moderate defaults this page reproduces <code>results/impact/impact_txn.json</code> (sha256 '+b.impact_json_sha256.slice(0,12)+
  ') for '+D.check.reproduces.length+' of its '+D.check.of+' agents, in all three brackets. Calls per GPU-hour: '+fmt0.format(D.calls_per_gpu_hour)+
  ' (measured, ROI.md §3). Simulator: ProcureSim, an in-memory procurement simulator; no customer deployment exists.';
}
buildFields();render();ratesTable();sensTable();foot();
})();
</script>
</body>
</html>
"""


def main() -> None:
    data = build()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__DATA__", blob))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data['agents'])} agents; reproduces impact_txn.json for "
          f"{len(data['check']['reproduces'])} of {data['check']['of']}")
    for a in data["agents"]:
        p = a["pooled"]
        print(f"  {a['label']}: VTC {p['VTC']['num']}/{p['VTC']['den']} FCR {p['FCR']['num']}/{p['FCR']['den']} "
              f"DMR {p['DMR_all']['num']}/{p['DMR_all']['den']}; moderate {a['sensitivity']['moderate']['faulted_mean']:,.0f} "
              f"per 1,000 faulted, {a['sensitivity']['moderate']['workflows']:,.0f} per 1,000 workflows")


if __name__ == "__main__":
    main()
