"""HTML rendering for the ActionProof product. Three information layers, each with one fixed treatment:
  visible (what the agent saw)  -- neutral ink on paper
  state   (what the database did) -- amber rule
  truth   (evaluator-only verdict) -- green or red rule, shown to the human user only
"""
from __future__ import annotations

import html
import json
from typing import Any

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--ink:#1d2329;--muted:#5b6670;--paper:#fbfaf7;--card:#ffffff;--line:#e3e0d8;--vis:#3a4a5a;--state:#b7791f;
--ok:#2f7d4f;--bad:#b3392f;--focus:#1f5f9e}
@media (prefers-color-scheme: dark){:root{--ink:#e8e6e1;--muted:#a6adb4;--paper:#16191c;--card:#1d2125;--line:#30363b;
--vis:#9fb0c2;--state:#e0a644;--ok:#5fbf86;--bad:#e2776c;--focus:#7fb2e5}}
.ap{font-family:'IBM Plex Sans',system-ui,sans-serif;color:var(--ink);background:var(--paper);line-height:1.5}
.ap h1,.ap h2,.ap h3{font-family:'Source Serif 4',Georgia,serif;font-weight:650;letter-spacing:-0.01em;margin:0 0 .4em}
.ap h1{font-size:2rem}.ap h2{font-size:1.35rem}.ap h3{font-size:1.05rem}
.ap .lede{font-size:1.05rem;color:var(--muted);max-width:52rem}
.ap .legend{display:flex;gap:1.2rem;flex-wrap:wrap;font-size:.85rem;color:var(--muted);margin:.6rem 0 1rem}
.ap .sw{display:inline-block;width:.8rem;height:.8rem;border-radius:2px;margin-right:.35rem;vertical-align:-1px}
.ap .row{display:grid;grid-template-columns:2.2rem 1fr 1fr 1fr;gap:.75rem;padding:.7rem .2rem;border-top:1px solid var(--line)}
.ap .row:first-child{border-top:none}
.ap .n{font-family:'IBM Plex Mono',monospace;color:var(--muted);font-size:.85rem;padding-top:.15rem}
.ap .cell{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:.55rem .65rem;font-size:.86rem;min-width:0}
.ap .vis{border-left:4px solid var(--vis)}.ap .state{border-left:4px solid var(--state)}
.ap .truth-ok{border-left:4px solid var(--ok)}.ap .state-neutral{border-left:4px solid var(--line)}.ap .truth-bad{border-left:4px solid var(--bad)}
.ap .lbl{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:.2rem}
.ap code,.ap pre{font-family:'IBM Plex Mono',monospace;font-size:.78rem;white-space:pre-wrap;word-break:break-word;margin:0}
.ap .flag{display:inline-block;font-weight:600;color:var(--bad);font-size:.8rem;margin-top:.25rem}
.ap .scripted{opacity:.7}
.ap table{border-collapse:collapse;width:100%;font-size:.88rem}
.ap th,.ap td{border-bottom:1px solid var(--line);padding:.45rem .5rem;text-align:left;vertical-align:top}
.ap th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.ap td.num{font-family:'IBM Plex Mono',monospace;text-align:right;white-space:nowrap}
.ap .tag{display:inline-block;font-size:.72rem;border:1px solid var(--line);border-radius:3px;padding:0 .35rem;color:var(--muted)}
.ap .tag.conf{border-color:var(--ok);color:var(--ok)}
.ap .note{font-size:.85rem;color:var(--muted);border-left:3px solid var(--line);padding-left:.7rem;margin:.8rem 0}
.ap .grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(20rem,1fr));gap:1rem}
.ap .ci{display:block;height:6px;background:var(--line);border-radius:3px;position:relative;margin-top:.3rem}
.ap .ci i{position:absolute;top:0;bottom:0;background:var(--focus);border-radius:3px}
.ap .ci b{position:absolute;top:-3px;width:2px;height:12px;background:var(--ink)}
@media (max-width:720px){.ap .row{grid-template-columns:1.6rem 1fr}.ap .row .cell{grid-column:2}}
"""

LEGEND = ('<div class="legend"><span><span class="sw" style="background:var(--vis)"></span>What the agent saw</span>'
          '<span><span class="sw" style="background:var(--state)"></span>What the database did</span>'
          '<span><span class="sw" style="background:var(--ok)"></span><span class="sw" style="background:var(--bad)"></span>'
          'Evaluator-only verdict (never shown to a policy)</span></div>')


def esc(x: Any) -> str:
    return html.escape(x if isinstance(x, str) else json.dumps(x, separators=(", ", ": ")))


def _call(c: dict) -> str:
    return f"<code>{esc(c['tool'])}({esc(c['args'])})</code>"


def timeline_html(rows: list[dict], title: str = "") -> str:
    out = [f'<div class="ap">{f"<h3>{esc(title)}</h3>" if title else ""}{LEGEND}']
    for r in rows:
        v, s, t = r["visible"], r["state"], r["truth"]
        diff = "".join(f"<div><b>{esc(k)}</b> +{len(rows_)}: <code>{esc(rows_)}</code></div>" for k, rows_ in s["new_rows"].items()) \
            or "<div>no change</div>"
        commit = (f"<div class='lbl'>commit: {esc(s['commit'])}{' · fault: ' + esc(s['fault']) if s['fault'] else ''}</div>"
                  if s["commit"] else "<div class='lbl'>no write</div>")
        verdicts = ", ".join(f"{k}: {v_}" for k, v_ in t["verdicts"].items())
        bad = t["diverged"]
        cls = "truth-bad" if bad else ("truth-ok" if all(x == "VERIFIED_SUCCESS" for x in t["verdicts"].values()) else "state-neutral")
        out.append(
            f'<div class="row{" scripted" if r.get("scripted") else ""}"><div class="n">{r["i"] + 1}</div>'
            f'<div class="cell vis"><div class="lbl">call{" (scripted prefix)" if r.get("scripted") else ""}</div>{_call(v["call"])}'
            f'<div class="lbl" style="margin-top:.35rem">response</div><code>{esc(v["response"])}</code></div>'
            f'<div class="cell state">{commit}{diff}</div>'
            f'<div class="cell {cls}"><div class="lbl">verdict after this call</div>{esc(verdicts)}'
            f'{"<div class=flag>Acknowledgement disagrees with the database</div>" if bad else ""}</div></div>')
    out.append("</div>")
    return "".join(out)


def rewards_html(rw: dict) -> str:
    ev = rw["evidence"]
    rows = "".join(f"<tr><td>{esc(k)}</td><td class='num'>{v:+.2f}</td></tr>" for k, v in rw.items() if k != "evidence")
    return (f'<div class="ap"><table><tr><th>Reward or metric</th><th>Value</th></tr>{rows}</table>'
            f'<div class="note">R_V reads the final database (step verdicts: {esc(ev["step_verdicts"])}). R_A reads only the last '
            f'acknowledgement of each requested write ({esc(ev["last_acknowledgements"])}). Report: {esc(ev["report"])}. '
            f'False completion: {ev["false_completion"]}; duplicate: {ev["duplicate"]}; collateral: {ev["collateral"]}.</div></div>')


def _ci_bar(est: float, lo: float, hi: float, span: float = 0.6) -> str:
    f = lambda x: max(0.0, min(100.0, (x + span) / (2 * span) * 100))
    return f'<span class="ci" title="95% interval"><i style="left:{f(lo):.1f}%;width:{f(hi) - f(lo):.1f}%"></i><b style="left:{f(0):.1f}%"></b></span>'


def dashboard_html(d: dict) -> str:
    rows = []
    for r in d["rows"]:
        tag = '<span class="tag conf">registered primary</span>' if r["status"] == "confirmatory" else '<span class="tag">secondary</span>'
        rows.append(f"<tr><td>{esc(r['comparison'])}<br>{tag}</td><td>{esc(r['split'])}</td><td>{esc(r['metric'])}</td>"
                    f"<td class='num'>{r['est']:+.3f}<br><small>[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]</small>{_ci_bar(r['est'], *r['ci95'])}</td>"
                    f"<td class='num'>{' '.join(f'{x:+.2f}' for x in r['per_seed'])}</td>"
                    f"<td class='num'>{r['n_seeds']} seeds<br>p₁={r['p_one_sided']:.3g}<br><small>{esc(r.get('p_label', 'sign-flip'))}</small></td><td><code>{esc(r['registration'])}</code></td></tr>")
    wf = d["by_workflow"]
    wrows = "".join(f"<tr><td>{esc(k.split('__')[0])}</td><td>{esc(k.split('__')[1])}</td>"
                    + "".join(f"<td class='num'>{v[w]['k']}/{v[w]['n']} = {v[w]['rate']:.2f}</td>" for w in ("order_prepay", "receive_pay", "pay_post"))
                    + "</tr>" for k, v in wf.items())
    limits = "".join(f"<li>{esc(x)}</li>" for x in d["limits"])
    return (f'<div class="ap"><h2>Registered results</h2><p class="lede">Differences in verified safe completion (VSC), false '
            f'completion and violations (in ProcureSim, almost all are duplicate payments left unrefunded). Intervals are paired task bootstraps with one task draw shared across '
            f'seeds; seed-level evidence is the per-seed column and the p under it (exact sign-flip: smallest 1/32 at five '
            f'seeds, 1/8 at three; unpaired seed permutation: smallest 1/252).</p><table><tr><th>Comparison</th><th>Split</th><th>Metric</th><th>Seed-avg Δ [95% CI]</th>'
            f'<th>Per seed</th><th>Seeds</th><th>Source</th></tr>{"".join(rows)}</table>'
            f'<h2 style="margin-top:1.5rem">Partial and duplicate recovery by workflow</h2><table><tr><th>Model</th><th>Split</th>'
            f'<th>Order → pay</th><th>Receive → pay</th><th>Pay first</th></tr>{wrows}</table>'
            f'<h2 style="margin-top:1.5rem">Limits</h2><ul>{limits}</ul>'
            f'<div class="note">Generated from {esc(", ".join(d["sources"].values()))} by <code>python -m product.build_data</code>.</div></div>')


def comparison_html(task: dict, models: dict) -> str:
    cols = []
    for m, run in task["runs"].items():
        s = run["score"]
        calls = "".join(f"<li class='{'scripted' if x.get('by') != 'agent' else ''}'><code>{esc(x['tool'])}</code> → "
                        f"<code>{esc(x['obs'].get('status'))}</code></li>" for x in run["trace"])
        verdict = "verified complete" if s["vsc"] else ("false completion" if s["false_success"] else "not completed")
        cls = "truth-ok" if s["vsc"] else "truth-bad"
        cols.append(f'<div class="cell {cls}"><h3>{esc(models[m])}</h3><div class="lbl">{esc(verdict)} · '
                    f'{run["policy_calls"]} calls · violation: {run["violation"]}</div><ol>{calls}</ol>'
                    f'<div class="lbl">report</div><code>{esc(run["report"])}</code><div class="lbl" style="margin-top:.3rem">'
                    f'final step verdicts</div><code>{esc(s["steps"])}</code></div>')
    return (f'<div class="ap"><p class="lede"><b>Task</b> {esc(task["task_id"])} · fault class <b>{esc(task["fault_class"])}</b> · '
            f'workflow <b>{esc(task["workflow"])}</b></p><p>{esc(task["prompt"])}</p><div class="grid2">{"".join(cols)}</div>'
            f'<div class="note">Recorded greedy evaluation episodes from the S3 confirmatory run '
            f'(<code>results/s3/conf/*__s3_conf_full.jsonl</code>); not live inference.</div></div>')
