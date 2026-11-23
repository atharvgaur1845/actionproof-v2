/* ---------- 06-08: retention, judges, ROI (data: S.upgrade, built by scripts/site_upgrade.py) ---------- */
{{ROIJS}}
(function(){const U=S.upgrade;if(!U)return;
 const pp=x=>x==null?"—":(x>=0?"+":"−")+Math.abs(x*100).toFixed(1);
 const cell=x=>{if(!x)return `<td>—</td>`;const cls=x.ci[1]<0?"dn":(x.ci[0]>0?"up":"");
  return `<td class="${cls}">${pp(x.d)}<small>[${pp(x.ci[0])}, ${pp(x.ci[1])}] · from ${(x.ref*100).toFixed(1)}</small></td>`};
 // 06 retention
 const r=U.r1;
 if(r.status!=="done"){$("#u-r1").innerHTML=`<tr><td>Running: generation and scoring of 33 models. This table fills from <span class="mono">${esc(r.source)}</span> when it exists.</td></tr>`;}
 else{$("#u-r1").innerHTML=`<tr><th>Model, change in points vs its start</th>${r.eps.map(e=>`<th>${esc(e[1])}</th>`).join("")}</tr>`+
  `<tr><td>Supervised warm start vs base model</td>${r.eps.map(e=>cell(r.warm[e[0]])).join("")}</tr>`+
  r.arms.map(a=>`<tr><td>${esc(a.label)} <small style="color:var(--gr)">${a.seeds} runs vs warm start</small></td>${r.eps.map(e=>cell(a.eps[e[0]])).join("")}</tr>`).join("");
  const v=r.verdicts||{},yes=b=>b?"supported":"not supported";
  const nm=k=>k.replace("a12_rv","state reward").replace("rsc","recovery start").replace(":"," · ").replace("ifeval_prompt_strict","IFEval").replace("bfcl_nonlive_ast","BFCL");
  const h2=v.H2?Object.entries(v.H2).map(([k,x])=>`${nm(k)}: ${x.noninferior?"no loss beyond 3 points":"<b>possible loss</b>"}`).join("<br>"):"pending";
  $("#u-r1v").innerHTML=`<div><div class="cap">H1 · does supervised warm-up cost ability?</div><p>${v.H1?Object.entries(v.H1).map(([k,x])=>`${k.startsWith("ifeval")?"IFEval":"BFCL"}: ${pp(x.delta)} points, ${yes(x.supported)}`).join("<br>"):"pending"}</p></div>`+
   `<div><div class="cap">H2 · does RL cost more than 3 points? (primary)</div><p>${h2}</p></div>`+
   `<div><div class="cap">H3 · does it stop abstaining when no tool fits?</div><p>${v.H3?Object.entries(v.H3).map(([k,x])=>`${k==="rsc"?"recovery start":"state reward"}: ${pp(x.delta)} points, ${yes(x.supported)}`).join("<br>"):"pending"}</p></div>`;}
 $("#u-r1-src").innerHTML=`Source: <b>${esc(r.source)}</b> · scripts/r1_stats.py · official checkers: bfcl-eval 2026.3.23, lm-eval 0.4.13 IFEval. Red: the whole interval is below zero.`;
 const t=U.r1t;
 $("#u-r1t").innerHTML=t.status==="done"?`<b>τ-bench retail (R1-T, descriptive).</b> ${Object.entries(t.contrasts).map(([k,x])=>`${k.replace(/_/g," ")}: ${pp(x.delta)} points [${pp(x.ci95[0])}, ${pp(x.ci95[1])}]`).join(" · ")}. Base pass^1 ${t.models.base?(t.models.base.pass1*100).toFixed(1)+"%":"—"}, context overflow in ${t.models.base?(t.models.base.overflow_rate*100).toFixed(0)+"%":"—"} of episodes. Simulated user: Aya Expanse 32B, so not comparable with published τ-bench numbers. <span class="mono">${esc(t.source)}</span>`
  :t.status==="pilot"?`<b>τ-bench retail (R1-T).</b> Feasibility pilot done: the base model overflowed its context in ${(t.pilot.overflow_rate*100).toFixed(0)}% of 20 episodes (the main run is cancelled above 50%). Main run in progress.`
  :`<b>τ-bench retail (R1-T).</b> Running: a feasibility pilot decides first whether an 8,192-token model can be tested at all (cancelled if more than half the episodes overflow).`;
 // 07 judges
 const j=U.j1;
 if(j.status!=="done"){$("#u-j1").innerHTML=`<tr><td>Running: 4 judges × 660 episodes × 2 views. Fills from <span class="mono">${esc(j.source)}</span>.</td></tr>`;}
 else{$("#u-j1").innerHTML=`<tr><th>Judge</th><th>Correct: transcript only</th><th>Answered (did not abstain)</th><th>Says "it happened" when it did not and nothing was checked</th><th>Correct: with database snapshot</th><th>$ per 1,000</th></tr>`+
  j.rows.map(x=>`<tr><td>${esc(x.judge)}</td><td>${x.transcript_acc==null?"—":pct(x.transcript_acc)}</td><td>${x.transcript_cov==null?"—":pct(x.transcript_cov)}</td><td class="${x.fooled>=0.25?"dn":""}">${x.fooled==null?"—":pct(x.fooled)}<small>of ${x.fooled_n} episodes</small></td><td>${x.state_acc==null?"—":pct(x.state_acc)}</td><td>${x.cost_per_1000==null?"—":"$"+x.cost_per_1000.toFixed(2)}</td></tr>`).join("");}
 if(j.status==="done"){const a1=j.addendum1;
  $("#u-j1v").innerHTML=`<div><div class="cap">Registered · J-H1 ${j.verdicts["J-H1"]?"supported":"not supported"}</div><p>Told that an "ok" reply is not proof, no judge said "it happened" on more than ${j.max_fooled} of ${j.n_unverifiable} unverifiable failures. They abstained instead, so they cannot certify outcomes either.${a1?" Without that one sentence (addendum 1, exploratory): "+Object.entries(a1).map(([k,v])=>v.v0?`${esc(k)} ${v.v0.fooled}/${v.v0.n_unverifiable}`:"").filter(Boolean).join(", ")+" fooled.":""}</p></div>`+
   `<div><div class="cap">Registered · J-H2 ${j.verdicts["J-H2"]?"supported":"not supported"}</div><p>Even with a database snapshot, two of four families stay near 89% correct, and a majority vote of all four reaches ${pct(j.state_committee)}. Majority voting is not ground truth.</p></div>`+
   `<div><div class="cap">Systematic bias</div><p>In ${j.all_four_wrong_state} snapshot cases all four judges contradict the verifier, and the verifier is right: a refunded duplicate (payment, duplicate, refund) nets to one payment, but every judge read it as "paid twice".</p></div>`;}
 $("#u-j1-src").innerHTML=`Source: <b>${esc(j.source)}</b> · evaluation/llm_judge_committee/analyze.py · correct = agrees with the verifier; an abstention counts as not correct.${j.human?` Human calibration: ${j.human.n} cases, one annotator.`:" Human calibration labels: pending."}`;
 // 08 ROI calculator
 const A=U.roi,SC=A.assumptions,F=[["fault_rate","Faults per workflow (%)",100],["value","Transaction value ($)",1],["exception_cost","Manual fix ($)",1],
  ["late_multiplier","Late-discovery multiplier",1],["dup_recovery","Duplicate recovered (%)",100],["gpu_hour","GPU hour ($)",1]];
 let cur={...SC.moderate},scn="moderate";
 const ctl=$("#u-roictl");
 ctl.innerHTML=F.map(([k,l])=>`<label>${l}<input type="number" step="any" min="0" data-k="${k}"></label>`).join("")+
  `<div class="roiscn" role="group" aria-label="Scenario">${Object.keys(SC).map(s=>`<button data-s="${s}">${s}</button>`).join("")}</div>`;
 const sync=()=>{F.forEach(([k,,m])=>{$(`[data-k="${k}"]`,ctl).value=+(cur[k]*m).toFixed(4)});$$(".roiscn button",ctl).forEach(b=>b.setAttribute("aria-pressed",b.dataset.s===scn))};
 const draw=()=>{const rows=Object.entries(A.arms).map(([n,a])=>[n,apRoiProject(a,cur,A.episodes_per_gpu_hour)]);
  const base=rows[0][1].cost_per_1000_ops.total_usd,mx=Math.max(...rows.map(r=>r[1].cost_per_1000_ops.total_usd));
  const usd=v=>"$"+Math.round(v).toLocaleString("en-US");
  $("#u-roi").innerHTML=`<tr><th>Per 1,000 workflows</th><th>False success</th><th>Duplicates left unrefunded</th><th>Sent to a human</th><th>Projected cost</th><th>vs SFT agent</th></tr>`+
   rows.map(([n,p])=>{const c=p.cost_per_1000_ops.total_usd,q=p.per_1000_ops;return `<tr><td>${esc(n)}</td><td>${q.false_success.toFixed(2)}</td><td>${q.unrefunded_duplicate.toFixed(2)}</td><td>${q.flagged.toFixed(2)}</td><td>${usd(c)}<span class="bar ${c>base*0.5?"r":""}" style="width:${Math.max(1,100*c/mx)}%"></span></td><td>${n===rows[0][0]?"—":(c<=base?"saves ":"costs ")+usd(Math.abs(base-c))}</td></tr>`}).join("");};
 ctl.addEventListener("input",e=>{const k=e.target.dataset.k;if(!k)return;const m=F.find(f=>f[0]===k)[2];cur[k]=Math.max(0,+e.target.value||0)/m;scn="custom";$$(".roiscn button",ctl).forEach(b=>b.setAttribute("aria-pressed","false"));draw()});
 ctl.addEventListener("click",e=>{const s=e.target.dataset&&e.target.dataset.s;if(!s)return;cur={...SC[s]};scn=s;sync();draw()});
 sync();draw();
 $("#u-roi-src").innerHTML=`Measured rates: <b>${esc(A.source)}</b> (s3_conf_full, every seed pooled; scripted rows on the same tasks). Formula: <span class="mono">demo/roi.js</span>, identical to scripts/roi_model.py (tested). Inference priced at the measured ${A.episodes_per_gpu_hour.toLocaleString("en-US")} policy calls per GPU-hour.`;
})();
