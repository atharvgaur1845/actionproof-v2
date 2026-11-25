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
/* ---------- 10: theory (data: S.upgrade.theory, scripts/site_upgrade.py::theory) ---------- */
(function(){const T=S.upgrade&&S.upgrade.theory;if(!T||T.status!=="done"||!document.getElementById("th-cost"))return;
 const COL={grpo:"#F05236",dr:"#2B6CB0"};const NS="http://www.w3.org/2000/svg";
 $("#th-leg").innerHTML=`<span><i style="background:${COL.grpo}"></i>GRPO (divides by the group's spread)</span><span><i style="background:${COL.dr}"></i>Dr. GRPO (subtracts the mean only)</span>`;
 const W=260,H=190,L=34,R=8,TP=22,B=34,cs=[...new Set(T.cost.map(r=>r.c))].sort((a,b)=>a-b);
 const lx=c=>L+(Math.log(c)-Math.log(cs[0]))/(Math.log(cs[cs.length-1])-Math.log(cs[0]))*(W-L-R),ly=v=>TP+(1-v)*(H-TP-B);
 const step=(pts,k)=>pts.map((r,i)=>i?`H${((lx(pts[i-1].c)+lx(r.c))/2).toFixed(1)}V${ly(r[k]).toFixed(1)}`:`M${lx(r.c).toFixed(1)},${ly(r[k]).toFixed(1)}`).join("")+`H${lx(pts[pts.length-1].c).toFixed(1)}`;
 $("#th-cost").innerHTML=[0.3,0.6].map(p=>{const pts=T.cost.filter(r=>r.p===p).sort((a,b)=>a.c-b.c);
  const xb=lx(Math.min(p,cs[cs.length-1]));
  const ticks=[0.01,0.1,1].filter(c=>c>=cs[0]&&c<=cs[cs.length-1]);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Learned read probability against read cost at p = ${p}">`+
   `<rect class="band" x="${L}" y="${TP}" width="${(xb-L).toFixed(1)}" height="${H-TP-B}"/>`+
   [0,0.5,1].map(v=>`<line class="grid" x1="${L}" x2="${W-R}" y1="${ly(v)}" y2="${ly(v)}"/><text x="${L-6}" y="${ly(v)+4}" text-anchor="end">${v}</text>`).join("")+
   ticks.map(c=>`<text x="${lx(c)}" y="${H-B+16}" text-anchor="middle">${c}</text>`).join("")+
   `<text class="ttl" x="${L}" y="12">p = ${p}</text><text x="${(L+W-R)/2}" y="${H-4}" text-anchor="middle">read cost (log)</text>`+
   `<path d="${step(pts,"drgrpo")}" fill="none" stroke="${COL.dr}" stroke-width="2" stroke-linejoin="round"/>`+
   `<path d="${step(pts,"grpo")}" fill="none" stroke="${COL.grpo}" stroke-width="2" stroke-dasharray="${p===0.6?"6 4":"0"}" stroke-linejoin="round"/>`+
   `</svg>`}).join("");
 const S2=$("#th-th1"),rows=T.th1,X0=210,X1=540,top=12,rh=29,x=v=>X0+v*(X1-X0);
 let h=[0,0.5,1].map(v=>`<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${top-4}" y2="${top+rows.length*rh}"/><text x="${x(v)}" y="${top+rows.length*rh+16}" text-anchor="middle">${v===1?"1.0":v}</text>`).join("");
 h+=rows.map((r,i)=>{const y=top+i*rh+rh/2,c=r.group_std?COL.grpo:COL.dr;
  return `<g class="row" tabindex="0" data-i="${i}"><rect x="0" y="${y-rh/2}" width="${X1+16}" height="${rh}" fill="transparent"/>`+
   `<text x="${X0-12}" y="${y+4}" text-anchor="end">${esc(r.name)}</text><line x1="${x(0)}" x2="${x(r.agreement)}" y1="${y}" y2="${y}" stroke="${c}" stroke-opacity=".35" stroke-width="2"/>`+
   `<circle cx="${x(r.agreement)}" cy="${y}" r="6" fill="${c}" stroke="var(--paper)" stroke-width="2"/></g>`}).join("");
 S2.innerHTML=h;const tip=$("#th-tip");
 const show=i=>{const r=rows[i];tip.textContent=`${r.name}: right decision in ${(r.agreement*100).toFixed(1)}% of problems · median regret ${r.regret.toFixed(3)}${r.group_std?" · divides by the group's spread":""}`};
 S2.querySelectorAll(".row").forEach(g=>{g.addEventListener("mouseenter",()=>show(+g.dataset.i));g.addEventListener("focus",()=>show(+g.dataset.i))});
 S2.addEventListener("mouseleave",()=>tip.textContent="");
 $("#th-src").innerHTML=`Source: <b>${esc(T.source)}</b> (generated on Slurm by scripts/theory/site_data.py). Left: ${esc(T.labels.cost)}. Right: ${esc(T.labels.th1)}; PREREGISTRATION_TH1.md, results/theory/th1_heldout_summary.json. Red dots: estimators that divide by the group's reward spread. Blue: estimators that do not. GRPO never reads at p = 0.3, even when reading is cheap. At p = 0.6 it always reads, even when reading costs more than the fault (dashed red). Dr. GRPO switches where the cost equals p. Proofs: docs/theory/PROOF_AND_ASSUMPTIONS.md.`})();

/* ---------- 09b: ledger and guard (data: S.upgrade.p2, scripts/site_upgrade.py::p2) ---------- */
(function(){const P=S.upgrade&&S.upgrade.p2;if(!P||P.status!=="done"||!document.getElementById("u-p2"))return;
 const pp=x=>x==null?"—":(x>=0?"+":"−")+Math.abs(x*100).toFixed(1),pc=x=>x==null?"—":(x*100).toFixed(1)+"%";
 const L={"E - D":"E (ledger in training) vs D","E - A":"E (ledger in training) vs A","B2 - A":"Ledger at run time vs A","Ag - A":"Guard on A","Cg - C":"Guard on C","Dg - D":"Guard on D","Eg - E":"Guard on E"};
 $("#u-p2").innerHTML=`<tr><th>Registered test</th><th>Verified success, sampled</th><th>Change</th><th>Holm p</th><th>Verdict</th></tr>`+
  P.rows.map(r=>`<tr><td><b>${r.id}</b> ${esc(L[r.contrast]||r.contrast)}${r.family==="secondary"?" <small>secondary</small>":""}</td><td>${pc(r.Y)} → ${pc(r.X)}</td><td class="${r.ci95[0]>0?"up":(r.ci95[1]<0?"dn":"")}">${pp(r.delta)}<small>[${pp(r.ci95[0])}, ${pp(r.ci95[1])}] · seeds ${r.seed_deltas.map(pp).join(" ")}</small></td><td>${r.holm_p.toFixed(4)}</td><td>${r.supported?"<b>supported</b>":"not supported"}</td></tr>`).join("");
 const R=P.references||{};
 $("#u-p2-src").innerHTML=`Source: <b>${esc(P.source)}</b> · scripts/p2_stats.py · Holm within each family. References on the same tasks: hand-written checker ${pc(R.O&&R.O.mean_vsc)} (greedy), class-matched control ${pc(R.M&&R.M.mean_vsc)}, warm start ${pc(R.W&&R.W.mean_vsc)}.${P.oracle_pp_blocked!=null?` Why the guard hurts (descriptive, after results): its read-before-retry rule would block the checker's own refund-then-re-pay repair on ${P.oracle_pp_blocked} of ${P.oracle_pp_tasks} short-payment tasks, and blocked agents rarely recover. results/quality/p2_guard_diagnosis.json`:""}`})();

/* ---------- 09c: P3 no-std on D, plus the exploratory 2x2 (data: S.upgrade.p3, scripts/site_upgrade.py::p3) ---------- */
(function(){const P=S.upgrade&&S.upgrade.p3;if(!P||!document.getElementById("u-p3"))return;
 if(P.status!=="done"&&!P.x2)return;document.getElementById("u-p3wrap").hidden=false;
 const pp=x=>x==null?"—":(x>=0?"+":"−")+Math.abs(x*100).toFixed(1),pc=x=>x==null?"—":(x*100).toFixed(1)+"%";
 const cls=c=>c[0]>0?"up":(c[1]<0?"dn":"");let h=`<tr><th>Test</th><th>Verified success, sampled</th><th>Change</th><th>Verdict</th></tr>`;
 if(P.status==="done")h+=`<tr><td><b>H1</b> Dn (no spread division) vs D</td><td>${pc(P.D)} → ${pc(P.Dn)}</td><td class="${cls(P.ci95)}">${pp(P.delta)}<small>[${pp(P.ci95[0])}, ${pp(P.ci95[1])}] · one-sided p ${P.p.toFixed(4)} · seeds ${P.seed_deltas.map(pp).join(" ")}</small></td><td>${P.supported?"<b>supported</b>":"not supported"}</td></tr>`;
 if(P.x2)h+=P.x2.rows.filter(r=>!/reference/.test(r.name)).map(r=>`<tr><td>${esc(r.name)} <small>exploratory</small></td><td>—</td><td class="${cls(r.ci95)}">${pp(r.delta)}<small>[${pp(r.ci95[0])}, ${pp(r.ci95[1])}]</small></td><td>descriptive</td></tr>`).join("");
 $("#u-p3").innerHTML=h;
 $("#u-p3-src").innerHTML=`Source: <b>${esc(P.source)}</b> · scripts/p3_stats.py · supported only if one-sided p < 0.05 and all three seed differences are positive.`+(P.x2?` Exploratory rows: <b>${esc(P.x2.source)}</b> (scripts/audit/dwarm_2x2.py; read rule fixed before output in research_audit/NOTE_DWARM_EVAL.md). W = warm start, A = recovery-start GRPO, Dw = recovery-SFT warm start of D, never evaluated in P1.`:"")})();

/* ---------- 09: practical agent (data: S.upgrade.p1, scripts/site_upgrade.py::p1) ---------- */
(function(){const P=S.upgrade&&S.upgrade.p1;if(!P)return;
 const pp=x=>x==null?"—":(x>=0?"+":"−")+Math.abs(x*100).toFixed(1);
 const pc=x=>x==null?"—":(x*100).toFixed(1)+"%",nm=x=>x==null?"—":x.toFixed(2);
 const SPL=[["p1_test_full","Full workflows"],["p1_test_pp","Short payment, unseen fractions"],["p1_test_rec","Mid-task recovery"],["p1_test_wf4","Unseen workflow"],["p1_test_combo","Two faults"]];
 if(P.status!=="done"){
  const d=P.dev&&P.dev.mean_sampled_vsc;
  $("#u-p1").innerHTML=`<tr><td>Running: arms C and D are training (3 seeds each); then every arm runs the 450 frozen test tasks once. Fills from <span class="mono">${esc(P.source)}</span>.${d?` Development (not the test): ${Object.entries(d).map(([a,v])=>`${a} ${pc(v)}`).join(" · ")}.`:""}</td></tr>`;
  $("#u-p1-src").textContent="";return;}
 const T=P.test,U=P.union;
 $("#u-p1").innerHTML=`<tr><th>Arm (3 seeds each)</th><th>Verified success, sampled</th><th>Change vs A</th><th>Holm p</th><th>Registered test</th>${SPL.map(s=>`<th>${s[1]}</th>`).join("")}</tr>`+
  Object.entries(P.arms).map(([a,lab])=>{const t=T[a];const cls=t?(t.ci95[0]>0?"up":(t.ci95[1]<0?"dn":"")):"";
   return `<tr><td><b>${a}</b> ${esc(lab)}</td><td>${pc(U[a].vsc.sampled)}<small>greedy ${pc(U[a].vsc.greedy)}</small></td>`+
   `<td class="${cls}">${t?`${pp(t.delta)}<small>[${pp(t.ci95[0])}, ${pp(t.ci95[1])}] · seeds ${t.seed_deltas.map(pp).join(" ")}</small>`:"reference"}</td>`+
   `<td>${t?t.holm_p.toFixed(4):"—"}</td><td>${t?(t.supported?"<b>supported</b>":"not supported"):"—"}</td>`+
   SPL.map(s=>`<td>${pc(P.splits[a][s[0]])}</td>`).join("")+`</tr>`}).join("");
 $("#u-p1-src").innerHTML=`Source: <b>${esc(P.source)}</b> · scripts/p1_stats.py. Change: paired task bootstrap on seed-averaged task scores, one-sided, Holm over B, C and D. Supported only if Holm p &lt; 0.05 <i>and</i> all three seed pairs improve. Development winner (labels the demo agent only): <b>${P.dev&&P.dev.winner?esc(P.dev.winner):"none beat A by 3 points"}</b>.`;
 const M=[["false_success","Reported done, but not done"],["duplicate_unrefunded","Duplicate left unrefunded"],["partial_pay_repaired","Short payments repaired"],["new_key_share_on_repay","Re-pays with a new key"],["redundant_reads_per_episode","Repeated reads per task"],["calls_per_success","Tool calls per success"],["context_ended","Ran out of context"]];
 $("#u-p1b").innerHTML=`<tr><th>Behaviour, sampled, all 450 tasks</th>${Object.keys(P.arms).map(a=>`<th>${a}</th>`).join("")}</tr>`+
  M.map(([k,l])=>`<tr><td>${l}</td>${Object.keys(P.arms).map(a=>{const v=U[a][k].sampled;return `<td>${k.endsWith("_episode")||k==="calls_per_success"?nm(v):pc(v)}</td>`}).join("")}</tr>`).join("");
 // same-task viewer: every task of the shown splits, in id order; no selection by outcome
 const ids=Object.keys(P.tasks),box=$("#u-p1x");
 const dis=id=>{const v=Object.values(P.tasks[id].arms).map(e=>e.vsc);return v.some(x=>x)&&v.some(x=>!x)};
 box.innerHTML=`<div class="top"><span class="tag rec">Recorded episodes · frozen test · seed 0 · greedy</span><span class="cap">No live model runs on this page</span></div>`+
  `<div class="p1top"><label class="cap" for="p1sel">Task</label><select id="p1sel"></select><div class="agents" role="group" aria-label="Filter"><button data-f="all" aria-pressed="true">All ${ids.length}</button><button data-f="dis" aria-pressed="false">Arms disagree (${ids.filter(dis).length})</button></div></div>`+
  `<div id="p1task" class="p1task"></div><div class="p1grid" id="p1grid"></div>`+
  `<p class="src" style="margin-top:12px">Each column is one arm on the same task. <b>Tool said</b> is what the agent saw. <b>Database</b> is the simulator's commit record, which the agent never sees. The verdict comes from the SQLite verifier. Tasks: ${P.show.map(esc).join(", ")}, every task listed, in id order.</p>`;
 const sel=$("#p1sel",box);let filt="all";
 const show=id=>{if(!id){$("#p1grid",box).innerHTML="";return}const t=P.tasks[id];
  $("#p1task",box).innerHTML=`${esc(t.prompt)}<small>${esc(t.workflow)} · injected fault (hidden from the agent): ${esc(t.faults.join(" + "))}</small>`;
  $("#p1grid",box).innerHTML=Object.keys(P.arms).map(a=>{const e=t.arms[a];if(!e)return `<div class="p1col"><h4>${a}</h4>—</div>`;
   const steps=e.steps.map(([tool,args,ack,db,pre])=>`<div class="p1s${pre?" pre":""}"><b>${esc(tool)}</b>${args?" "+esc(args):""}${pre?" · scripted start":""}<span class="ak">tool said: ${esc(ack)}</span>${db?`<span class="db${db.startsWith("full")?"":" bad"}">database: ${esc(db)}</span>`:""}</div>`).join("");
   const v=Object.entries(e.verdicts).map(([k,x])=>`${k} ${String(x).replace("VERIFIED_","").toLowerCase().replace(/_/g," ")}`).join(" · ");
   return `<div class="p1col"><h4>${a} · ${esc(P.arms[a])}</h4>${steps}<div class="p1v ${e.vsc?"ok":"no"}">Verifier: ${e.vsc?"verified success":"not done"}${e.false_success?" · reported done falsely":""}${e.violation?" · duplicate left":""}<br><small>${esc(v)} · report ${esc(e.report||"none")} · ended by ${esc(e.ended_by)}</small></div></div>`}).join("")};
 const fill=()=>{const list=filt==="dis"?ids.filter(dis):ids;sel.innerHTML=list.map(id=>`<option value="${id}">${id} · ${esc(P.tasks[id].faults.join(" + "))}${dis(id)?" · arms disagree":""}</option>`).join("");show(list[0])};
 sel.addEventListener("change",()=>show(sel.value));
 $$(".agents button",box).forEach(b=>b.addEventListener("click",()=>{filt=b.dataset.f;$$(".agents button",box).forEach(x=>x.setAttribute("aria-pressed",x===b));fill()}));
 fill();
})();
