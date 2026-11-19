"""ActionProof gateway process (FastAPI). It alone holds the simulated services, contracts, verifier, recovery and the
evidence ledger; agents in other processes reach the services only through it (SDK: `actionproof.integrity.sdk`).

    actionproof serve --test-mode --port 8765        # dashboard at http://127.0.0.1:8765/

Endpoints
  POST /v1/transactions                 bind a contract for a task (test mode: any ProcureSim task, optional faults)
  GET  /v1/transactions/{tx}/tools      tool schemas the agent may call (same names and signatures as the services)
  POST /v1/transactions/{tx}/calls      one tool call through the gateway
  POST /v1/transactions/{tx}/finalize   the gateway's disposition (the agent's claim is recorded, never trusted)
  GET  /v1/transactions[/{tx}]          list / detail (contract, steps, disposition, decisions, truth score)
  GET  /v1/transactions/{tx}/events     ledger events of one transaction
  GET  /v1/stream                       server-sent events: every ledger append, live
  GET  /v1/ledger/verify                recompute the hash chain
  POST /v1/test/runs                    test mode: run a scripted or model agent on a task, in one mode
  POST /v1/test/compare                 test mode: the same task, agent and faults under every mode (real executions)
  POST /v1/test/tamper                  test mode: overwrite one stored event (to demonstrate detection)
  GET  /v1/scenarios                    test mode: tasks available for the playground
  GET  /v1/evaluation                   result files behind the evaluation page (missing files are reported as pending)
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from actionproof.bench.generator_proc import SPLITS, load_tasks
from actionproof.integrity.contracts import Contract
from actionproof.integrity.gateway import MODES, ProcGateway
from actionproof.integrity.ledger import EvidenceLedger
from actionproof.integrity.sim.agents import POLICIES
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc
from actionproof.integrity.sim.truth import score

ROOT = Path(__file__).resolve().parents[2]
DASH = Path(__file__).resolve().parent / "dashboard"
PUBLIC_SPLITS = ("proc_dev", "proc_val", "proc_heldout_service")  # never serve confirmatory splits in the playground
EVAL_FILES = {
    "h7_matrix": "results/a11/h7_matrix.json",
    "reward_truth_table": "results/a12/reward_truth_table.json",
    "a12_gate": "results/a12/gate.json",
    "a12_h3_predictions": "results/a12/h3_predictions.json",
    "a12_summary": "results/s2/a12_summary.json",
    "a11_summary": "results/s2/a11_summary.json",
    "h4_summary": "results/s2/h4_summary.json",
    "stage1_confirmatory": "results/confirmatory.json",
}


class TxIn(BaseModel):
    task_id: str
    split: str = "proc_dev"
    mode: str = "enforce"
    faults: list[dict[str, Any]] | None = None
    never_commit: str | None = None


class CallIn(BaseModel):
    tool: str
    args: dict[str, Any] = {}


class RunIn(TxIn):
    agent: str = "naive"


class TamperIn(BaseModel):
    seq: int


class State:
    def __init__(self, ledger_path: str | None, test_mode: bool) -> None:
        self.ledger = EvidenceLedger(ledger_path)
        self.test_mode = test_mode
        self.tx: dict[str, dict[str, Any]] = {}
        self.tasks = {s: {t.task_id: t for t in load_tasks(ROOT / f"data/bench/proc/{s}.jsonl")} for s in PUBLIC_SPLITS}
        self.lock = threading.Lock()
        self.listeners: list[queue.Queue] = []
        self.ledger.subscribers.append(self._fan_out)

    def _fan_out(self, e: Any) -> None:
        for q in list(self.listeners):
            q.put(e.to_json())


def _task(st: State, body: TxIn) -> Any:
    if body.split not in st.tasks or body.task_id not in st.tasks[body.split]:
        raise HTTPException(404, f"unknown task {body.split}/{body.task_id}")
    t = copy.deepcopy(st.tasks[body.split][body.task_id])
    if body.faults is not None or body.never_commit is not None:
        t.meta.update(faults=body.faults or [], never_commit=body.never_commit)
    return t


def _open(st: State, body: TxIn) -> str:
    if body.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    t = _task(st, body)
    tx = f"tx-{uuid.uuid4().hex[:10]}"
    env = make_env_proc(t, tx)
    gw = ProcGateway(env, Contract.for_task(t), body.mode, ledger=st.ledger, tx_id=tx)
    st.tx[tx] = {"task": t, "env": env, "gw": gw, "mode": body.mode, "agent": None, "status": "open"}
    return tx


def _disposition(st: State, tx: str) -> dict[str, Any] | None:
    """A finalized transaction's disposition as recorded at finalization. Recomputing it re-verifies every step and
    appends evidence events, so a read (GET) would otherwise grow the append-only ledger."""
    r = st.tx[tx]
    if r["mode"] == "passthrough":
        return None
    if r.get("final") is None:
        if r["status"] != "finalized" and not r["env"].done:
            return r["gw"].disposition()
        fin = [e.body for e in st.ledger.events(tx) if e.kind == "finalized"]
        r["final"] = {k: v for k, v in fin[-1].items() if k != "ledger_head"} if fin else r["gw"].disposition()
    return r["final"]


def _detail(st: State, tx: str) -> dict[str, Any]:
    if tx not in st.tx:
        raise HTTPException(404, "unknown transaction")
    r = st.tx[tx]
    t, env, gw = r["task"], r["env"], r["gw"]
    d = _disposition(st, tx) if r["mode"] != "passthrough" else {"status": (env.final_report or {}).get("status")}
    rec = episode_record_proc(t, env)
    sysrec = score({**rec, "report": {"status": d["status"], "summary": ""}})
    return {"tx_id": tx, "task_id": t.task_id, "prompt": t.prompt, "workflow": t.workflow, "mode": r["mode"],
            "agent": r["agent"], "fault_class": t.meta["fault_class"], "faults": t.meta["faults"],
            "never_commit": t.meta.get("never_commit"), "contract": gw.contract.model_dump(mode="json"),
            "contract_hash": gw.contract.content_hash(), "disposition": d, "decisions": gw.decisions,
            "agent_view": gw.agent_view, "agent_report": env.final_report,
            "truth": {"agent_level": rec["score"], "system_level": sysrec},
            "status": r["status"]}


def create_app(ledger_path: str | None = None, test_mode: bool = True) -> FastAPI:
    st = State(ledger_path, test_mode)
    app = FastAPI(title="ActionProof gateway", version="0.2.0")
    app.state.ap = st

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(DASH / "index.html")

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        p = (DASH / name).resolve()
        if p.parent != DASH.resolve() or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.post("/v1/transactions")
    def open_tx(body: TxIn) -> dict[str, Any]:
        with st.lock:
            tx = _open(st, body)
        return {"tx_id": tx, "contract_hash": st.tx[tx]["gw"].contract.content_hash()}

    @app.get("/v1/transactions/{tx}/tools")
    def tools(tx: str) -> dict[str, Any]:
        from actionproof.rollout.lite_format import build_schema
        if tx not in st.tx:
            raise HTTPException(404)
        gw = st.tx[tx]["gw"]
        try:
            schemas = [build_schema(f) for f in gw.tools()]
        except Exception:  # transformers missing: names and docstrings only
            schemas = [{"name": f.__name__, "description": (f.__doc__ or "").strip()} for f in gw.tools()]
        return {"system_prompt": gw.adapt_prompt(""), "user_prompt": st.tx[tx]["task"].prompt, "tools": schemas}

    @app.post("/v1/transactions/{tx}/calls")
    def call(tx: str, body: CallIn) -> dict[str, Any]:
        if tx not in st.tx:
            raise HTTPException(404)
        with st.lock:
            return st.tx[tx]["gw"].call(body.tool, body.args)

    @app.post("/v1/transactions/{tx}/finalize")
    def finalize(tx: str) -> dict[str, Any]:
        if tx not in st.tx:
            raise HTTPException(404)
        with st.lock:
            st.tx[tx]["status"] = "finalized"
            st.tx[tx]["final"] = st.tx[tx]["gw"].finalize()
            return st.tx[tx]["final"]

    @app.get("/v1/transactions")
    def list_tx() -> list[dict[str, Any]]:
        out = []
        for tx, r in reversed(list(st.tx.items())):
            out.append({"tx_id": tx, "task_id": r["task"].task_id, "mode": r["mode"], "agent": r["agent"],
                        "status": r["status"], "fault_class": r["task"].meta["fault_class"],
                        "disposition": _disposition(st, tx)["status"] if r["mode"] != "passthrough" else None})
        return out

    @app.get("/v1/transactions/{tx}")
    def detail(tx: str) -> dict[str, Any]:
        with st.lock:
            return _detail(st, tx)

    @app.get("/v1/transactions/{tx}/events")
    def events(tx: str) -> list[dict[str, Any]]:
        return [e.to_json() for e in st.ledger.events(tx)]

    @app.get("/v1/stream")
    async def stream(after: int = 0) -> StreamingResponse:
        q: queue.Queue = queue.Queue()
        st.listeners.append(q)

        async def gen():
            try:
                for e in st.ledger.events(after=after):  # replay what the client missed
                    yield f"id: {e.seq}\ndata: {json.dumps(e.to_json())}\n\n"
                while True:
                    try:
                        e = q.get_nowait()
                        yield f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        await asyncio.sleep(0.25)
            finally:
                st.listeners.remove(q)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/v1/ledger/verify")
    def verify() -> dict[str, Any]:
        return st.ledger.verify()

    def _require_test() -> None:
        if not st.test_mode:
            raise HTTPException(403, "test mode is off")

    @app.get("/v1/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        _require_test()
        return [{"split": s, "task_id": t.task_id, "workflow": t.workflow, "fault_class": t.meta["fault_class"],
                 "prompt": t.prompt, "requested": t.meta["requested"]} for s in PUBLIC_SPLITS
                for t in list(st.tasks[s].values())[:40]]

    def _run(body: RunIn) -> str:
        if body.agent not in POLICIES and body.agent != "llm":
            raise HTTPException(400, f"agent must be one of {sorted(POLICIES)} or llm")
        with st.lock:
            tx = _open(st, body)
            st.tx[tx]["agent"] = body.agent
            r = st.tx[tx]
            if body.agent == "llm":
                from actionproof.integrity.llm_agent import run_llm_agent
                run_llm_agent(r["gw"], r["task"])
            else:
                POLICIES[body.agent](r["gw"], r["task"])
            if not r["env"].done:
                r["gw"].finalize()
            r["status"] = "finalized"
            _disposition(st, tx)  # record the finalized disposition now, before any tamper demonstration
        return tx

    @app.post("/v1/test/runs")
    def run(body: RunIn) -> dict[str, Any]:
        _require_test()
        return _detail(st, _run(body))

    @app.post("/v1/test/compare")
    def compare(body: RunIn) -> dict[str, Any]:
        _require_test()
        rows = []
        for mode in ("passthrough", "observe", "enforce", "enforce_inform"):
            d = _detail(st, _run(RunIn(**{**body.model_dump(), "mode": mode})))
            sysl, agl = d["truth"]["system_level"], d["truth"]["agent_level"]
            calls = len(d["agent_view"])
            rows.append({"mode": mode, "tx_id": d["tx_id"], "verified_completion": sysl["vsc"],
                         "false_finalization": int(sysl["false_success"]),
                         "agent_false_claim": int(agl["false_success"]),
                         "duplicate_payment": int(sysl["steps"].get("pay") == "DUPLICATE"),
                         "disposition": d["disposition"]["status"],
                         "exposure_cents": d["disposition"].get("exposure_cents"), "agent_calls": calls,
                         "recovery_actions": [x["action"] for x in d["decisions"]]})
        return {"task_id": body.task_id, "agent": body.agent, "faults": body.faults, "rows": rows}

    @app.post("/v1/test/tamper")
    def tamper(body: TamperIn) -> dict[str, Any]:
        _require_test()
        st.ledger.tamper(body.seq, {"tampered": True, "note": "edited after the fact"})
        return st.ledger.verify()

    @app.get("/v1/evaluation")
    def evaluation() -> dict[str, Any]:
        out = {}
        for k, rel in EVAL_FILES.items():
            p = ROOT / rel
            out[k] = {"path": rel, "status": "available", "data": json.loads(p.read_text())} if p.exists() else \
                {"path": rel, "status": "pending"}
        return out

    return app


app = create_app(os.environ.get("AP_LEDGER"), os.environ.get("AP_TEST_MODE", "1") == "1")
