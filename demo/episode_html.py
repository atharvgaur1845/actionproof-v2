"""Step-card timeline for recorded episodes (no Gradio import; used at build time by scripts/build_demo_data.py).

Each recorded episode is replayed step by step on a fresh simulator (every observation must match the record), so
each card can show what the tool CLAIMED next to what the DATABASE holds, plus the database after every step.
The policies emit tool calls only (no free-text reasoning). The "thinking trail" is therefore what the model planned
in each turn (all calls it wrote out) against what was executed.
"""
from __future__ import annotations

import html
import json
import re

from actionproof.env.db import snapshot
from actionproof.env.tools_ecom import TOOL_META, VERIFICATION_TOOLS
from actionproof.rollout.agenttune_rollout import make_env

ENTITY_ARGS = ("order_id", "ticket_id", "sku")
TOOL_RE = re.compile(r'"tool_name"\s*:\s*"([A-Za-z_]+)"')
RESULT_RE = re.compile(r"Tool result for ")
e = html.escape


def _ids(args) -> set:
    a = args if isinstance(args, dict) else {}
    return {a.get(k) for k in ENTITY_ARGS} - {None}


def _turns(rec: dict) -> list[tuple[list[str], int]]:
    """Per assistant turn: (tool names it planned, number of calls executed from it)."""
    conv = rec.get("conversation") or []
    out = []
    for i, m in enumerate(conv):
        if m["role"] != "assistant":
            continue
        planned = TOOL_RE.findall(m["content"])
        nxt = conv[i + 1]["content"] if i + 1 < len(conv) and conv[i + 1]["role"] == "user" else ""
        out.append((planned, len(RESULT_RE.findall(nxt))))
    return out


def _db_state(snap: dict, ids: dict) -> dict:
    o, t = ids.get("order"), ids.get("ticket")
    order = next((r for r in snap["orders"] if r["order_id"] == o), {})
    ticket = next((r for r in snap["tickets"] if r.get("ticket_id") == t or r.get("order_id") == o), {})
    return {"refund": any(r["order_id"] == o for r in snap["refunds"]), "order": order.get("status", "?"),
            "ticket": ticket.get("status", "?"), "esc": len(snap["escalations"])}


def _ack(obs) -> str:
    if not isinstance(obs, dict):
        return str(obs)[:60]
    st = obs.get("status", "?")
    extra = next((f"{k} {obs[k]}" for k in ("refund_id", "order_status", "ticket_status", "error", "eligibility")
                  if k in obs), "")
    return (f"{st} · {extra}" if extra else st)[:60]


def _evidence(tool: str, obs) -> str | None:
    if not isinstance(obs, dict):
        return None
    if tool == "query_ledger":
        refs = obs.get("refunds")
        if refs is not None:
            return "ledger shows <code>refunds: []</code> → refund missing" if not refs else \
                f"ledger shows refund {e(str(refs[0].get('refund_id', '')))} → present"
    if tool == "lookup_order" and isinstance(obs.get("order"), dict):
        return f"order status = <b>{e(str(obs['order'].get('status')))}</b>"
    if tool == "get_refund_status":
        return f"refund status = <b>{e(str(obs.get('refund_status', obs.get('state', '?'))))}</b>"
    if tool == "get_ticket" and isinstance(obs.get("ticket"), dict):
        return f"ticket status = <b>{e(str(obs['ticket'].get('status')))}</b>"
    return None


def build_lane(task, rec: dict, score_fn) -> dict:
    """Replay `rec` step by step; return steps, database track, summary and rendered HTML."""
    env = make_env(task, rec.get("rollout_id", "replay"), seed=0)
    ids = task.meta.get("ids", {})
    turns = _turns(rec)
    turn_of, planned_after, k = {}, {}, 0  # model-step index -> turn number; first step of turn -> unexecuted plan
    model_idx = [i for i, s in enumerate(rec["trace"]) if not s.get("by")]
    for t, (planned, n_exec) in enumerate(turns):
        n_exec = max(n_exec, 1) if planned else n_exec
        if k < len(model_idx):
            planned_after[model_idx[k]] = (planned, n_exec)
        for _ in range(n_exec):
            if k < len(model_idx):
                turn_of[model_idx[k]] = t + 1
                k += 1
    steps, track, written, seen_writes = [], [], set(), {}
    for i, s in enumerate(rec["trace"]):
        obs = env.call(s["tool"], s["args"])
        if obs != s["obs"]:
            raise AssertionError(f"{task.task_id}: replay diverged at step {s['idx']}")
        att = next((a for a in env.log.attempts if a.action_idx == s["idx"]), None)
        kind = TOOL_META.get(s["tool"], (None, "read"))[1]
        st = {"i": i, "tool": s["tool"], "args": s["args"], "obs": s["obs"], "harness": bool(s.get("by")),
              "turn": turn_of.get(i), "kind": kind, "ack": _ack(s["obs"])}
        if i in planned_after:
            planned, n_exec = planned_after[i]
            st["plan"] = {"n": len(planned), "executed": n_exec, "ghost": planned[n_exec:]}
        if kind == "read" and s["tool"] in VERIFICATION_TOOLS and _ids(s["args"]) & written and not s.get("by"):
            st["kind"], st["evidence"] = "verify", _evidence(s["tool"], s["obs"])
        if att is not None:
            st["fault"] = att.fault
            st["db"] = ("partial" if att.fault == "F3_partial" else "written" if att.committed else "not_written")
            key = (att.op, att.entity)
            seen_writes[key] = seen_writes.get(key, 0) + 1
            if seen_writes[key] > 1:
                prev = [x for x in steps if not x["harness"]]
                st["retry"] = {"n": seen_writes[key] - 1, "after_evidence": bool(prev) and prev[-1]["kind"] == "verify"}
            written |= _ids(s["args"])
        if s["tool"] == "escalate_to_human":
            st["kind"] = "escalate"
        steps.append(st)
        track.append(_db_state(snapshot(env.conn), ids))
    sc = score_fn(task, env)
    v = sc["verdict"]
    rep = env.final_report or {}
    conv = rec.get("conversation") or []
    summary = {"turns": len(turns), "calls": len(model_idx),
               "chars": sum(len(m["content"]) for m in conv if m["role"] == "assistant"),
               "n_verify": sum(x["kind"] == "verify" for x in steps), "vsc": int(sc["VSC"]),
               "verdict": {k2: getattr(v, k2) for k2 in ("G", "P", "V", "I", "D", "F_over", "F_under")},
               "rewards": sc["rewards"], "report": {"status": rep.get("status", "none"), "summary": rep.get("summary", "")}}
    return {"steps": steps, "db_track": track, "summary": summary,
            "raw": {k2: rec.get(k2) for k2 in ("trace", "attempts", "receipts", "verdict", "report")}}


# ---------------------------------------------------------------- HTML
ICON = {"read": "◔", "verify": "⌕", "write": "✎", "escalate": "⇪", "terminal": "■"}


def _chip(cls: str, text: str, title: str = "") -> str:
    return f'<span class="chip {cls}" title="{e(title)}">{text}</span>'


def _args(args) -> str:
    if not isinstance(args, dict):
        return ""
    return " · ".join(f"{e(k)}={e(str(v))[:40]}" for k, v in args.items() if k not in ("summary", "text"))


def card(st: dict) -> str:
    if st["tool"] == "submit_report":
        return ""
    cls = "harness" if st["harness"] else st["kind"]
    badges = []
    if st.get("retry"):
        r = st["retry"]
        badges.append(_chip("good" if r["after_evidence"] else "bad",
                            f"RETRY #{r['n']} · {'after evidence' if r['after_evidence'] else 'blind'}"))
    if st["kind"] == "verify":
        badges.append(_chip("verify", "READ-BACK"))
    if st["kind"] == "escalate":
        badges.append(_chip("esc", "ESCALATED"))
    if st.get("fault") == "F4_timeout":
        badges.append(_chip("partial", "TRAP: blind retry would duplicate"))
    plan = ""
    if st.get("plan") and st["plan"]["n"] > 1:
        p = st["plan"]
        ghosts = "".join(_chip("ghost", e(g)) for g in p["ghost"])
        plan = (f'<div class="plan">turn {st["turn"]} · model wrote {p["n"]} calls · executed {p["executed"]}'
                f'{" · not executed: " + ghosts if ghosts else ""}</div>')
    truth = ""
    if "db" in st:
        ack_ok = st["ack"].startswith("success")
        dbc = {"written": ("good", "✓ WRITTEN"), "partial": ("partial", "◐ PARTIAL: refund row only"),
               "not_written": ("bad hatch", "✗ NOT WRITTEN")}[st["db"]]
        fault = f' · {e(st["fault"])}' if st.get("fault") else ""
        mismatch = st["db"] != "written" and ack_ok
        truth = (f'<div class="truth"><div class="claim"><span class="lab">tool claimed</span>'
                 f'{_chip("good" if ack_ok else "muted", e(st["ack"]))}</div>'
                 f'<div class="arrow{" mm" if mismatch else ""}">{"≠ mismatch" if mismatch else "→"}</div>'
                 f'<div class="claim"><span class="lab">database holds</span>{_chip(dbc[0], dbc[1] + fault)}</div></div>')
    else:
        truth = f'<div class="obs">{_chip("muted", e(st["ack"]))}</div>'
    ev = f'<div class="evidence">{st["evidence"]}</div>' if st.get("evidence") else ""
    raw = (f'<details class="raw"><summary>raw</summary><pre>{e(json.dumps({"args": st["args"], "obs": st["obs"]}, indent=1))}'
           f'</pre></details>')
    who = '<span class="who">harness</span>' if st["harness"] else ""
    return (f'<div class="card {cls}">{plan}<div class="head"><span class="ic">{ICON.get(st["kind"], "◔")}</span>'
            f'<b>{e(st["tool"])}</b> <span class="args">{_args(st["args"])}</span>{who} {"".join(badges)}</div>'
            f'{truth}{ev}{raw}</div>')


def _bar(v: float, cls: str = "") -> str:
    w = max(0.0, min(1.0, float(v))) * 100
    return f'<span class="mbar {cls}"><i style="width:{w:.0f}%"></i></span>'


def final_card(lane: dict) -> str:
    sm = lane["summary"]
    v, rep = sm["verdict"], sm["report"]
    if sm["vsc"]:
        ban = ("good", "✓ Verified safe completion")
    elif v["F_over"]:
        ban = ("bad", "✗ False completion: claimed done, the database disagrees")
    elif v["F_under"]:
        ban = ("partial", "◐ Task done, but reported as not done")
    else:
        ban = ("muted", "Not completed, reported honestly")
    st = rep["status"]
    pill = {"completed": "good", "failed": "partial", "escalated": "esc"}.get(st, "muted")
    terms = "".join(f'<div class="term"><span>{k}</span>{_bar(v[k] if k != "I" else 1 - v[k], "inv" if k in ("I", "D") else "")}'
                    f'<em>{v[k]:.2f}</em></div>' for k in ("G", "P", "V", "I", "D"))
    rw = "".join(f'<div class="term wide"><span>{e(k)}</span>{_bar((r + 1) / 2.3 if "PCSTR" in k else r)}<em>{r:.2f}</em></div>'
                 for k, r in sm["rewards"].items())
    summ = f'<q>{e(rep["summary"])}</q>' if rep.get("summary") else ""
    return (f'<div class="card final"><div class="banner {ban[0]}">{ban[1]}</div>'
            f'<div class="fgrid"><div><div class="lab">agent reported</div>{_chip(pill, e(st))} {summ}</div>'
            f'<div><div class="lab">verifier (database)</div>{terms}</div>'
            f'<div><div class="lab">what each reward paid</div>{rw}</div></div></div>')


def track_html(lane: dict) -> str:
    cells = []
    for j, t in enumerate(lane["db_track"]):
        cells.append(f'<span class="dbc {"on" if t["refund"] else "off"}" title="after step {j + 1}: refund row '
                     f'{"present" if t["refund"] else "absent"} · order {e(str(t["order"]))} · ticket {e(str(t["ticket"]))}'
                     f' · escalations {t["esc"]}"></span>')
    return f'<div class="track"><span class="lab">refund in database, step by step</span>{"".join(cells)}</div>'


def lane_header(name: str, lane: dict, channel: str) -> str:
    sm = lane["summary"]
    return (f'<div class="lhead {channel}"><div class="lname">{e(name)}</div>'
            f'<div class="lstats">{sm["turns"]} turns · {sm["calls"]} calls · {sm["n_verify"]} read-backs · '
            f'{sm["chars"]:,} chars generated</div>{track_html(lane)}</div>')


def grid(lanes: dict[str, dict], channels: dict[str, str]) -> str:
    """Rows are steps; every lane puts its k-th visible step in row k, so lanes stay aligned by step."""
    names = list(lanes)
    cols = [[card(s) for s in lanes[n]["steps"] if s["tool"] != "submit_report"] for n in names]
    rows = max(len(c) for c in cols)
    out = [f'<div class="ep-grid" style="--lanes:{len(names)}">', '<div class="gut"></div>']
    out += [lane_header(n, lanes[n], channels.get(n, "warm")) for n in names]
    for r in range(rows):
        out.append(f'<div class="gut">{r + 1:02d}</div>')
        out += [c[r] if r < len(c) else '<div class="card ended"></div>' for c in cols]
    out.append('<div class="gut">end</div>')
    out += [final_card(lanes[n]) for n in names]
    out.append("</div>")
    out.append('<details class="raw lane-raw"><summary>full raw records (trace, attempts, receipts, verdict)</summary>'
               + "".join(f"<h4>{e(n)}</h4><pre>{e(json.dumps(lanes[n]['raw'], indent=1)[:20000])}</pre>" for n in names)
               + "</details>")
    return "".join(out)
