"""ActionProof: verify what agents actually changed.

    python -m product.app            # http://127.0.0.1:7860

What is live and what is recorded:
  Incident simulator, Reward lab, Recovery arena  -- live, deterministic, CPU: the real ProcureSim simulator and verifier
  Policy comparison                                -- recorded greedy evaluation episodes (S3 confirmatory), not live inference
  Research dashboard                               -- generated from result files (`python -m product.build_data`)
No model is called, no external system is touched, and no real payment exists anywhere in this app.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from product import render
from product.engine import ACTION_LABELS, ACTIONS, FAULTS, WORKFLOWS, Arena, Scenario, run_policy

DATA = Path(__file__).resolve().parent / "data"
REPO = "https://github.com/atharvgaur1845/actionproof-v2"
FAULT_STEPS = {"order_prepay": ["order", "pay"], "receive_pay": ["receive", "pay"], "pay_post": ["pay"]}


def _load(name: str) -> dict | None:
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else None


def simulate(workflow, fault, step, ratio, seed, policy):
    try:
        sc = Scenario(workflow=workflow, fault=fault, fault_step=step, ratio=float(ratio), seed=int(seed))
        sc.validate()
    except AssertionError:
        msg = "This combination is not defined: partial commits apply to pay and receive only, and the faulted step must belong to the workflow."
        return f'<div class="ap"><p class="note">{msg}</p></div>', ""
    r = run_policy(sc, policy)
    title = f"{'Trusting agent (never reads back)' if policy == 'trusting' else 'Verifying reference agent'} · {WORKFLOWS[workflow]} · {FAULTS[fault]}"
    return render.timeline_html(r["timeline"], title), render.rewards_html(r["rewards"])


def arena_start(workflow, fault, step, ratio, seed):
    try:
        a = Arena(Scenario(workflow=workflow, fault=fault, fault_step=step, ratio=float(ratio), seed=int(seed)))
    except AssertionError:
        return None, '<div class="ap"><p class="note">This combination is not defined.</p></div>', ""
    from product.engine import timeline
    return a, render.timeline_html(timeline(a.task, a.env), "Scripted history up to the faulted write"), \
        '<div class="ap"><p class="note">Choose the next action. The acknowledgement above is all an agent would see.</p></div>'


def arena_act(a, action):
    if a is None:
        return a, '<div class="ap"><p class="note">Start a scenario first.</p></div>', ""
    if a.env.done:
        return a, gr.update(), '<div class="ap"><p class="note">The episode has ended. Start a new scenario.</p></div>'
    out = a.act(action)
    rw = render.rewards_html(out["rewards"]) if out["done"] else \
        '<div class="ap"><p class="note">Episode still open. Final scores appear after a report (continue past the last step, escalate or stop).</p></div>'
    return a, render.timeline_html(out["timeline"], f"After: {action}"), rw


def compare(idx):
    r = _load("replays.json")
    if not r:
        return '<div class="ap"><p class="note">Replay data not built: run <code>python -m product.build_data</code>.</p></div>'
    t = r["tasks"][int(idx)]
    return render.comparison_html(t, r["models"])


def replay_choices():
    r = _load("replays.json")
    return [(f"{t['fault_class']} · {t['workflow']} · {t['task_id']}", i) for i, t in enumerate(r["tasks"])] if r else []


INTRO = f"""<div class="ap"><h1>ActionProof</h1><p class="lede"><b>Verify what agents actually changed.</b> An agent says the
payment succeeded. ActionProof checks the database, reconstructs the execution history, and shows whether the agent made
the right recovery decision. In this simulator a write's acknowledgement can be byte-identical whether the write
committed, half-committed or did nothing.</p>{render.LEGEND}<p class="note">Live tabs run the real ProcureSim simulator and
verifier on this machine (CPU, deterministic, seeded). The policy comparison replays recorded evaluation episodes. Nothing
here calls a model or an external system. Code and data: <a href="{REPO}">{REPO}</a>.</p></div>"""

REPRO = f"""<div class="ap"><h2>Reproduce</h2><table>
<tr><th>Artefact</th><th>Location</th></tr>
<tr><td>Simulator and verifier</td><td><code>actionproof/integrity/sim/</code> (spec <code>docs/stage2/VERIFIER_SPEC.md</code>)</td></tr>
<tr><td>Registrations</td><td><code>PREREGISTRATION_A12*.md</code>, <code>PREREGISTRATION_S3.md</code>, <code>PREREGISTRATION_S3_ADDENDUM1.md</code></td></tr>
<tr><td>Recovery-start curriculum</td><td><code>actionproof/stage3/</code>; train: <code>python -m actionproof.stage3.train --reward rv --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl</code></td></tr>
<tr><td>Evaluation</td><td><code>python scripts/s3_eval.py --split s3_conf_full ...</code>; statistics <code>python scripts/s3_stats.py</code></td></tr>
<tr><td>Claims ledger</td><td><code>claims/claims.yaml</code>; check every number: <code>python scripts/s2_numcheck.py</code></td></tr>
<tr><td>Corrections</td><td><code>docs/CONSISTENCY_AUDIT.md</code>, <code>docs/stage2/DECISIONS.md</code></td></tr>
<tr><td>This app's data</td><td><code>python -m product.build_data</code> (from result files only)</td></tr>
</table><p class="note">Repository: <a href="{REPO}">{REPO}</a>. Model adapters and full traces are on the
project's Hugging Face collections listed in the README.</p></div>"""


def build() -> gr.Blocks:
    with gr.Blocks(title="ActionProof") as demo:
        gr.HTML(f"<style>{render.CSS}</style>" + INTRO)
        with gr.Tab("Incident simulator"):
            with gr.Row():
                wf = gr.Dropdown(list(WORKFLOWS), value="pay_post", label="Workflow")
                ft = gr.Dropdown(list(FAULTS), value="partial", label="Fault")
                st = gr.Dropdown(["pay", "receive", "order"], value="pay", label="Faulted step")
                rt = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Partial-commit ratio")
                sd = gr.Number(value=7, precision=0, label="Scenario seed")
                pol = gr.Radio([("Trusting agent", "trusting"), ("Verifying reference", "verifying")], value="trusting", label="Agent")
            go = gr.Button("Run scenario", variant="primary")
            tl, rw = gr.HTML(), gr.HTML()
            go.click(simulate, [wf, ft, st, rt, sd, pol], [tl, rw])
            gr.HTML('<div class="ap"><p class="note">The table under the timeline is the reward laboratory: the same '
                    'trajectory scored by the state reward, the acknowledgement reward and the exploratory safety-first reward.</p></div>')
        with gr.Tab("Recovery arena"):
            state = gr.State(None)
            with gr.Row():
                wf2 = gr.Dropdown(list(WORKFLOWS), value="pay_post", label="Workflow")
                ft2 = gr.Dropdown(list(FAULTS), value="partial", label="Fault")
                st2 = gr.Dropdown(["pay", "receive", "order"], value="pay", label="Faulted step")
                rt2 = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Partial-commit ratio")
                sd2 = gr.Number(value=7, precision=0, label="Scenario seed")
            start = gr.Button("Start scenario", variant="primary")
            with gr.Row():
                btns = [gr.Button(ACTION_LABELS[a]) for a in ACTIONS]
            atl, arw = gr.HTML(), gr.HTML()
            start.click(arena_start, [wf2, ft2, st2, rt2, sd2], [state, atl, arw])
            for b, a in zip(btns, ACTIONS):
                b.click(lambda s, a=a: arena_act(s, a), [state], [state, atl, arw])
        with gr.Tab("Policy comparison"):
            sel = gr.Dropdown(replay_choices(), value=0, label="Fresh confirmatory task (same initial state and fault for every model)")
            cmp_html = gr.HTML(compare(0) if replay_choices() else "")
            sel.change(compare, [sel], [cmp_html])
        with gr.Tab("Research dashboard"):
            d = _load("dashboard.json")
            gr.HTML(render.dashboard_html(d) if d else '<div class="ap"><p class="note">Run <code>python -m product.build_data</code>.</p></div>')
            if d and d.get("evidence"):
                gr.HTML(render.evidence_html(d["evidence"]))
        with gr.Tab("Reproducibility"):
            gr.HTML(REPRO)
    return demo


if __name__ == "__main__":
    build().launch()
