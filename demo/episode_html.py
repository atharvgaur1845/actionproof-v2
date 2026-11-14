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


# ---------------------------------------------------------------- HTML (reconciliation-ledger style)
def _args(args) -> str:
    if not isinstance(args, dict):
        return ""
    return " · ".join(f"{e(k)}={e(str(v))[:40]}" for k, v in args.items() if k not in ("summary", "text"))


def card(st: dict) -> str:
    if st["tool"] == "submit_report":
        return ""
    tags = []
    if st["kind"] == "verify":
        tags.append('<span class="tag ink">read-back</span>')
    if st.get("retry"):
        r = st["retry"]
        tags.append(f'<span class="tag {"ink" if r["after_evidence"] else "pen"}">retry {r["n"]}, '
                    f'{"after reading the ledger" if r["after_evidence"] else "without checking"}</span>')
    if st["kind"] == "escalate":
        tags.append('<span class="tag">escalated to a human</span>')
    if st["harness"]:
        tags.append('<span class="tag">harness</span>')
    plan = ""
    if st.get("plan") and st["plan"]["n"] > 1:
        p = st["plan"]
        plan = (f'<div class="plan" title="planned but not run: {e(", ".join(p["ghost"]))}">wrote {p["n"]} calls ahead, '
                f'{p["executed"]} ran</div>')
    if "db" in st:
        ack_ok = st["ack"].startswith("success")
        fault = f' ({e(st["fault"])})' if st.get("fault") else ""
        db = {"written": ('ok', "written"), "partial": ("pen", "half written: refund row only, order not updated"),
              "not_written": ("pen", "no row")}[st["db"]]
        contradicted = st["db"] != "written" and ack_ok
        body = (f'<div class="recon"><span class="k">tool said</span><span class="{"struck" if contradicted else ""}">'
                f'{e(st["ack"])}</span><span class="k">ledger holds</span><span class="{db[0]}">{db[1]}{fault}</span></div>')
    else:
        body = f'<div class="out">{e(st["ack"])}</div>'
    ev = f'<div class="evidence">{st["evidence"]}</div>' if st.get("evidence") else ""
    raw = (f'<details class="raw"><summary>Raw call</summary><pre>{e(json.dumps({"args": st["args"], "obs": st["obs"]}, indent=1))}'
           f'</pre></details>')
    return (f'<div class="card{" harness" if st["harness"] else ""}">{plan}<div class="row1"><span class="tool">{e(st["tool"])}</span>'
            f'<span class="args">{_args(st["args"])}</span>{"".join(tags)}</div>{body}{ev}{raw}</div>')


def _bar(v: float, lead: bool = False) -> str:
    w = max(0.0, min(1.0, float(v))) * 100
    return f'<span class="mbar{" lead" if lead else ""}"><i style="width:{w:.0f}%"></i></span>'


def final_card(lane: dict) -> str:
    sm = lane["summary"]
    v, rep = sm["verdict"], sm["report"]
    if sm["vsc"]:
        ban = ("good", "Verified: the database matches the report")
    elif v["F_over"]:
        ban = ("bad", "False completion: reported done, the database disagrees")
    elif v["F_under"]:
        ban = ("muted", "Done, but reported as not done")
    else:
        ban = ("muted", "Not completed, reported honestly")
    terms = "".join(f'<div class="term"><span>{k}</span>{_bar(v[k] if k not in ("I", "D") else 1 - v[k])}<em>{v[k]:.2f}</em></div>'
                    for k in ("G", "P", "V", "I", "D"))
    rw = "".join(f'<div class="term wide"><span>{e(k)}</span>{_bar((r + 1) / 2.3 if "PCSTR" in k else r, "PCSTR" in k)}'
                 f'<em>{r:.2f}</em></div>' for k, r in sm["rewards"].items())
    summ = f'<q>{e(rep["summary"])}</q>' if rep.get("summary") else ""
    return (f'<div class="card final"><div class="verdict {ban[0]}">{ban[1]}</div><div class="fgrid">'
            f'<div><div class="lab">Agent reported</div><span class="status">{e(rep["status"])}</span>{summ}</div>'
            f'<div><div class="lab">Verifier terms, from the database</div>{terms}</div>'
            f'<div><div class="lab">What each reward paid for this episode</div>{rw}</div></div></div>')


def track_html(lane: dict) -> str:
    cells = "".join(f'<span class="dbc {"on" if t["refund"] else ""}" title="after step {j + 1}: refund row '
                    f'{"present" if t["refund"] else "absent"}, order {e(str(t["order"]))}, ticket {e(str(t["ticket"]))}"></span>'
                    for j, t in enumerate(lane["db_track"]))
    return f'<div class="track"><span class="lab">Refund in ledger, per step</span>{cells}</div>'


def lane_header(name: str, lane: dict, channel: str) -> str:
    sm = lane["summary"]
    return (f'<div class="lhead {channel}"><div class="lname">{e(name)}</div>'
            f'<div class="lstats">{sm["calls"]} calls · {sm["n_verify"]} read-backs · {sm["chars"]:,} chars generated</div>'
            f'{track_html(lane)}</div>')


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
    out.append('<div class="gut"></div>')
    out += [final_card(lanes[n]) for n in names]
    out.append("</div>")
    out.append('<details class="raw lane-raw"><summary>Full raw records</summary>'
               + "".join(f"<h4>{e(n)}</h4><pre>{e(json.dumps(lanes[n]['raw'], indent=1)[:20000])}</pre>" for n in names)
               + "</details>")
    return "".join(out)
