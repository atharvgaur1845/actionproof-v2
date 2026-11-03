"""Demo helpers run without a GPU: fault injection, rendering, scoring."""
import importlib, sys, types
import pytest

gr = pytest.importorskip("gradio")


def test_fault_injection_and_render():
    sys.path.insert(0, "demo")
    app = importlib.import_module("app")
    t = app.build_task(0, "F1 phantom ack: says success, nothing written")
    env = app.make_env(t, "x", seed=0)
    ids = t.meta["ids"]
    env.call("issue_refund", {"order_id": ids["order"], "amount_cents": ids["amount"], "method": ids["method"]})
    env.call("submit_report", {"status": "completed", "summary": "done"})
    s = app.score(t, env)
    md = app.render(t, env, s)
    assert s["VSC"] == 0 and s["verdict"].F_over == 1 and "NOT written" in md and "False completion" in md
