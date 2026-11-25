"""App logic with no gradio / spaces / transformers imports, so it is unit-testable.

Streams are lists/generators of plain-dict events. Nothing hidden (fault plan, database rows, committed flags)
appears in any event before the final `end` event; the reveal is built from the scored record afterwards.
"""
from __future__ import annotations

import copy
import datetime as dt
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from actionproof_min.agents import naive, oracle
from actionproof_min.env import STEP_READ, STEP_TOOL, WRITE_TOOLS
from actionproof_min.env_s3 import make_env_s3
from actionproof_min.episode import MAX_NEW_TOKENS, MAX_TURNS, stream_episode
from actionproof_min.runtime import episode_record_proc
from actionproof_min.tasks import ProcTask, load_tasks

REPLAY_URL = "https://huggingface.co/spaces/atharvgaur18/actionproof"
BASE_REPO = "CohereLabs/tiny-aya-global"
TRAINED_REPO = "atharvgaur18/actionproof-agent-d"
MODELS = {"base": BASE_REPO, "trained": TRAINED_REPO}
SESSION_CAP = 10
DATA = Path(__file__).resolve().parent / "data" / "tasks.json"

# Fixed, deliberately not in the order a reader would guess; the key is shown only after a run.
REALITIES = {
    "Reality A": {"kind": "lost_ack", "name": "lost acknowledgement",
                  "what": "The payment was committed, but the reply was a timeout."},
    "Reality B": {"kind": None, "name": "committed",
                  "what": "The payment was committed and the reply said ok."},
    "Reality C": {"kind": "phantom", "name": "phantom",
                  "what": "The reply said ok, but nothing was committed."},
    "Reality D": {"kind": "partial", "name": "partial (ratio 0.5)",
                  "what": "The reply said ok, but only half the amount was committed."},
}
SCRIPTED = {"scripted_naive": ("SCRIPTED: acknowledgement-trusting", naive),
            "scripted_oracle": ("SCRIPTED: transaction-aware oracle", oracle)}
AGENTS = {"base": "Base model: tiny-aya-global (live)", "trained": "Trained agent (live)",
          "both": "Compare: base and trained (live)", **{k: v[0] for k, v in SCRIPTED.items()}}


# ------------------------------------------------------------------ tasks and realities
def load_curated() -> list[ProcTask]:
    return load_tasks(DATA)


def task_choices() -> dict[str, str]:
    """Label -> task_id. The label is the task's own request text, which is agent-visible."""
    out = {}
    for t in load_curated():
        out[f"{t.task_id.replace('p1_test_full-', 'Task ')} · {t.workflow.replace('_', ' ')}"] = t.task_id
    return out


def get_task(task_id: str) -> ProcTask:
    return next(t for t in load_curated() if t.task_id == task_id)


def apply_reality(task: ProcTask, reality: str) -> ProcTask:
    """Same request and world; `meta.faults` replaced by the reality's fault on the pay step's first call."""
    spec = REALITIES[reality]
    t = copy.deepcopy(task)
    f: list[dict[str, Any]] = []
    if spec["kind"] is not None:
        f = [{"tool": STEP_TOOL["pay"], "k": 0, "kind": spec["kind"]}]
        if spec["kind"] == "partial":
            f[0]["ratio"] = 0.5
    t.meta["faults"], t.meta["never_commit"] = f, None
    t.meta["fault_class"] = spec["kind"] or "clean"
    return t


# ------------------------------------------------------------------ event sources
def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def scripted_events(policy_key: str, task: ProcTask, reality: str) -> Iterator[dict]:
    """Run a scripted policy to completion, then replay its agent-visible calls as events."""
    label, policy = SCRIPTED[policy_key]
    t = apply_reality(task, reality)
    env = make_env_s3(t, f"live-{policy_key}-{task.task_id}")
    yield {"type": "start", "system": None, "prompt": t.prompt}
    policy(env, t)
    for s in env.trace:
        if s.get("by", "agent") == "agent":
            yield {"type": "call", "turn": s["idx"] + 1, "tool": s["tool"], "args": s["args"], "reply": s["obs"]}
    rec = episode_record_proc(t, env)
    yield {"type": "end", "ended_by": "report", "rec": rec, "turns": len(env.trace), "invalid": 0, "n0": 0}


def model_events(generate: Callable[[list[dict]], str], task: ProcTask, reality: str,
                 system_fn: Callable | None = None) -> Iterator[dict]:
    """The p1_eval episode loop on a reality of the task. `generate` is injectable."""
    t = apply_reality(task, reality)
    kw = {"system_fn": system_fn} if system_fn else {}
    yield from stream_episode(t, generate, f"live-{task.task_id}", **kw)


# ------------------------------------------------------------------ rendering (agent-visible only until `end`)
_ACK = re.compile(r'("status":\s*"ok")')


def _esc(s: Any) -> str:
    return html.escape(str(s))


def render_reply(obs: dict) -> str:
    out = html.escape(json.dumps(obs), quote=False)
    return out.replace('"status": "ok"', '<mark class="ack">"status": "ok"</mark>')


def render_event(ev: dict) -> str:
    t = ev["type"]
    if t == "call":
        args = json.dumps(ev["args"]) if ev["args"] is not None else "(invalid JSON)"
        is_report = ev["tool"] == "submit_report"
        cls = "step report" if is_report else "step"
        head = "Agent report" if is_report else f"Turn {ev['turn']}"
        return (f'<div class="{cls}"><div class="hd">{head}</div>'
                f'<div class="call"><span class="lbl">call</span> <code>{_esc(ev["tool"])}</code> '
                f'<code class="args">{html.escape(args, quote=False)}</code></div>'
                f'<div class="reply"><span class="lbl">reply</span> <code>{render_reply(ev["reply"])}</code></div></div>')
    if t == "text":
        raw = ev["raw"].strip()
        return (f'<div class="step raw"><div class="hd">Turn {ev["turn"]}: raw model output</div>'
                f'<pre>{html.escape(raw[:1200], quote=False)}</pre></div>')
    return ""


def banner(agent_key: str, model_id: str | None, stamp: str) -> str:
    if model_id:
        return (f'<div class="banner live">LIVE MODEL RUN · {_esc(model_id)} · {_esc(stamp)} · '
                f'greedy, max {MAX_TURNS} turns</div>')
    return f'<div class="banner scripted">{_esc(SCRIPTED[agent_key][0])} · no model, no GPU · {_esc(stamp)}</div>'


def ended_note(end: dict) -> str:
    why = {"report": "The agent submitted its report.", "no_call": "The agent replied without a tool call; the episode ended.",
           "max_turns": f"The agent used all {MAX_TURNS} turns without reporting.",
           "context": "The context limit was reached."}[end["ended_by"]]
    return f'<div class="note">{_esc(why)}</div>'


# ------------------------------------------------------------------ reveal (after the run only)
STEPS = ("order", "pay", "receive", "post")
TABLE_COLS = {"supplier_orders": ("order_ref", "sku", "qty", "status"),
              "payments": ("payment_id", "amount_cents", "payee_account", "kind", "refund_of"),
              "receipts": ("receipt_id", "qty", "kind"), "journal": ("entry_id", "amount_cents", "kind")}


def _table(title: str, cols: tuple[str, ...], rows: list[dict]) -> str:
    if not rows:
        return f'<div class="tbl"><div class="tt">{_esc(title)}</div><div class="empty">no rows</div></div>'
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_esc(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="tbl"><div class="tt">{_esc(title)}</div><table><tr>{head}</tr>{body}</table></div>'


def po_rows(rec: dict) -> dict[str, list[dict]]:
    pid = rec["task"]["meta"]["po_id"]
    return {t: [r for r in rec["final_state"].get(t, []) if r["po_id"] == pid] for t in TABLE_COLS}


def decision_record(rec: dict, reality: str) -> dict[str, Any]:
    """What the agent intended, acknowledged, read and did, next to what the verifier found."""
    task, meta = rec["task"], rec["task"]["meta"]
    pid = meta["po_id"]
    po = next(p for p in task["initial_state"]["pos"] if p["po_id"] == pid)
    agent = [s for s in rec["trace"] if s.get("by", "agent") == "agent"]
    at = {a["idx"]: a for a in rec["attempts"]}
    acks, reads, recovery, seen_writes = [], [], [], {}
    for i, s in enumerate(agent):
        tool = s["tool"]
        if tool in WRITE_TOOLS:
            a = at.get(s["idx"], {})
            acks.append({"tool": tool, "reply": s["obs"].get("status") if s["obs"].get("status") == "ok" else
                         s["obs"].get("error", s["obs"].get("status")), "committed": a.get("committed")})
            seen_writes[tool] = seen_writes.get(tool, 0) + 1
            if tool == "refund_payment":
                recovery.append(f"refund_payment {s['args'].get('payment_id')}")
            elif seen_writes[tool] > 1 and tool in STEP_TOOL.values():
                recovery.append(f"repeated {tool} (key {s['args'].get('idempotency_key')})")
            elif tool == "escalate_to_human":
                recovery.append("escalate_to_human")
            step = next((k for k, v in STEP_TOOL.items() if v == tool), None)
            if step:
                window = []
                for x in agent[i + 1:]:
                    if x["tool"] in WRITE_TOOLS or x["tool"] == "submit_report":
                        break
                    window.append(x)
                hit = any(x["tool"] == STEP_READ[step] and x["args"].get("po_id") == pid for x in window)
                reads.append({"after": tool, "read": STEP_READ[step] if hit else None})
    rep = rec.get("report") or {}
    sc = rec["score"]
    return {
        "intent": {"po_id": pid, "steps_requested": meta["requested"], "amount_cents": po["amount_cents"]},
        "postconditions": [f"order: one placed supplier order for qty {po['qty']}" if "order" in meta["requested"] else None,
                           f"pay: exactly one live payment of {po['amount_cents']} cents to the registered payee",
                           f"receive: net received quantity {po['qty']}" if "receive" in meta["requested"] else None,
                           f"post: one journal entry for {po['amount_cents']} cents"],
        "acks_seen": acks, "evidence_read": reads,
        "disposition": {"status": rep.get("status"), "summary": rep.get("summary")},
        "recovery_actions": recovery,
        "verification": {"step_verdicts": sc["steps"], "goal_met": sc["goal_met"], "vsc": bool(sc["vsc"]),
                         "false_success": bool(sc["false_success"])},
        "reality": reality,
    }


def render_decision_record(dr: dict) -> str:
    pcs = [p for p in dr["postconditions"] if p]
    lines = [("Intent", f"Process {dr['intent']['po_id']}: {', '.join(dr['intent']['steps_requested'])} "
                        f"({dr['intent']['amount_cents']} cents)"),
             ("Postconditions", "; ".join(pcs)),
             ("Acks seen", "; ".join(f"{a['tool']} -> {a['reply']}" for a in dr["acks_seen"]) or "none"),
             ("Evidence read", "; ".join(f"after {r['after']}: {r['read'] or 'no read before the next write or report'}"
                                         for r in dr["evidence_read"]) or "no writes"),
             ("Disposition", f"{dr['disposition']['status']}: {dr['disposition']['summary']}"),
             ("Recovery actions", "; ".join(dr["recovery_actions"]) or "none"),
             ("Verification result", ", ".join(f"{k}={v}" for k, v in dr["verification"]["step_verdicts"].items()))]
    rows = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in lines)
    return (f'<div class="tbl"><div class="tt">Transaction decision record</div><table class="kv">{rows}</table>'
            f'<div class="fine">A record of what this run did and what the verifier found. It is a description of one '
            f'run, not a guarantee about any other.</div></div>')


def render_reveal(rec: dict, reality: str, side: str = "") -> str:
    sc, spec = rec["score"], REALITIES[reality]
    verd = "".join(f"<tr><th>{_esc(s)}</th><td>{_esc(v)}</td></tr>" for s, v in sc["steps"].items())
    pay_att = [a for a in rec["attempts"] if a["tool"] == "pay_supplier" and a["po_id"] == rec["task"]["meta"]["po_id"]]
    att = "".join(f"<tr><td>{_esc(a['tool'])}</td><td>{_esc(a['obs_status'])}</td><td>{_esc(a['committed'])}</td></tr>"
                  for a in rec["attempts"] if a["po_id"] == rec["task"]["meta"]["po_id"])
    rows = po_rows(rec)
    summary = (f'<div class="verdict"><span>VSC <b>{int(sc["vsc"])}</b></span><span>False success '
               f'<b>{str(bool(sc["false_success"])).lower()}</b></span><span>Goal met <b>{str(sc["goal_met"]).lower()}</b></span></div>')
    key = (f'<div class="note"><b>{_esc(reality)} was: {_esc(spec["name"])}.</b> {_esc(spec["what"])} '
           f'The first pay call was '
           f'{"never made" if not pay_att else "answered " + _esc(pay_att[0]["obs_status"]) + ", committed: " + _esc(pay_att[0]["committed"])}.</div>')
    parts = [key, summary,
             '<div class="grid2">' + _table("payments (PO)", TABLE_COLS["payments"], rows["payments"]) +
             _table("receipts (PO)", TABLE_COLS["receipts"], rows["receipts"]) +
             _table("journal (PO)", TABLE_COLS["journal"], rows["journal"]) +
             _table("supplier orders (PO)", TABLE_COLS["supplier_orders"], rows["supplier_orders"]) + "</div>",
             f'<div class="tbl"><div class="tt">Verifier verdict per step</div><table class="kv">{verd}</table></div>',
             f'<div class="tbl"><div class="tt">Acknowledgement against effect, per write</div>'
             f'<table><tr><th>write</th><th>reply status</th><th>committed</th></tr>{att}</table></div>',
             render_decision_record(decision_record(rec, reality))]
    return "".join(parts)


# ------------------------------------------------------------------ session driver
def run_side(side: str, agent_key: str, task: ProcTask, reality: str,
             model_stream: Callable[[str, str, str], Iterator[dict]] | None) -> Iterator[tuple[str, str, dict | None]]:
    """Yield (banner_html, timeline_html, rec_or_None). `model_stream(model_key, task_id, reality)` is injectable."""
    stamp = utc_now()
    if agent_key in SCRIPTED:
        ban, events = banner(agent_key, None, stamp), scripted_events(agent_key, task, reality)
    else:
        ban, events = banner(agent_key, MODELS[agent_key], stamp), model_stream(agent_key, task.task_id, reality)
    t = apply_reality(task, reality)
    body = f'<div class="req"><div class="hd">Request</div>{_esc(t.prompt)}</div>'
    yield ban, body, None
    for ev in events:
        if ev["type"] == "end":
            body += ended_note(ev)
            yield ban, body, ev["rec"]
            return
        body += render_event(ev)
        yield ban, body, None
        if agent_key in SCRIPTED:
            time.sleep(0.35)


class Budget:
    """Per-session cap on live model runs (scripted runs are free)."""

    def __init__(self, cap: int = SESSION_CAP) -> None:
        self.cap = cap

    def charge(self, state: dict, n: int) -> bool:
        if state.get("runs", 0) + n > self.cap:
            return False
        state["runs"] = state.get("runs", 0) + n
        return True


def run_session(agent: str, task_id: str, reality: str, state: dict,
                model_stream: Callable[[str, str, str], Iterator[dict]] | None,
                budget: Budget | None = None) -> Iterator[dict]:
    """Yield UI dicts: {banner_l, left, banner_r, right, reveal, status}. Reveal is empty until every side has ended."""
    budget = budget or Budget()
    task = get_task(task_id)
    sides = ["base", "trained"] if agent == "both" else [agent]
    live = [s for s in sides if s not in SCRIPTED]
    ui = {"banner_l": "", "left": "", "banner_r": "", "right": "", "reveal": "", "status": ""}
    if live and not budget.charge(state, len(live)):
        ui["status"] = f"Session cap reached: {budget.cap} live model runs per session. Scripted runs are unlimited."
        yield dict(ui)
        return
    recs: dict[str, dict] = {}
    for i, s in enumerate(sides):
        bk, tk = ("banner_l", "left") if i == 0 else ("banner_r", "right")
        ui["status"] = f"Running {AGENTS[s]}"
        for ban, body, rec in run_side(s, s, task, reality, model_stream):
            ui[bk], ui[tk] = ban, body
            if rec is not None:
                recs[s] = rec
            yield dict(ui)
    parts = []
    for s in sides:
        head = f'<h4>{_esc(AGENTS[s])}</h4>' if len(sides) > 1 else ""
        parts.append(f'<div class="side">{head}{render_reveal(recs[s], reality)}</div>')
    ui["reveal"] = '<div class="cmp">' + "".join(parts) + "</div>"
    ui["status"] = "Run finished. Hidden state revealed below."
    yield dict(ui)
