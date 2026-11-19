"""`actionproof` command line.

  actionproof serve [--port 8765] [--ledger ledger.db] [--no-test-mode]
  actionproof run --task proc_dev-0007 [--split proc_dev] --mode enforce --agent naive [--fault pay_supplier:0=phantom ...]
  actionproof compare --task proc_dev-0007 --agent naive --fault pay_supplier:0=phantom
  actionproof ledger verify|anchor --ledger ledger.db [--anchors anchors.jsonl]
  actionproof contract show --task proc_dev-0007   (prints the bound contract and its hash)
  actionproof contract schema                       (JSON Schema of the contract format)
"""
from __future__ import annotations

import argparse
import json
import sys


def _faults(specs: list[str]) -> tuple[list[dict], str | None]:
    out, never = [], None
    for s in specs or []:
        lhs, kind = s.split("=")
        tool, k = lhs.split(":")
        if kind == "never_commit":
            never = tool
        else:
            out.append({"tool": tool, "k": int(k), "kind": kind})
    return out, never


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="actionproof")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--ledger", default=None)
    s.add_argument("--no-test-mode", action="store_true")
    for name in ("run", "compare"):
        r = sub.add_parser(name)
        r.add_argument("--task", required=True)
        r.add_argument("--split", default="proc_dev")
        r.add_argument("--mode", default="enforce")
        r.add_argument("--agent", default="naive")
        r.add_argument("--fault", action="append", default=[], help="tool:k=kind, e.g. pay_supplier:0=phantom")
    lg = sub.add_parser("ledger")
    lg.add_argument("action", choices=("verify", "anchor"))
    lg.add_argument("--ledger", required=True)
    lg.add_argument("--anchors", default=None)
    c = sub.add_parser("contract")
    c.add_argument("action", choices=("show", "schema"))
    c.add_argument("--task", default=None)
    c.add_argument("--split", default="proc_dev")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import os
        import uvicorn
        if a.ledger:
            os.environ["AP_LEDGER"] = a.ledger
        os.environ["AP_TEST_MODE"] = "0" if a.no_test_mode else "1"
        from actionproof.integrity.api import create_app
        uvicorn.run(create_app(a.ledger, not a.no_test_mode), host=a.host, port=a.port, log_level="warning")
    elif a.cmd in ("run", "compare"):
        from fastapi.testclient import TestClient
        from actionproof.integrity.api import create_app
        cl = TestClient(create_app(None, True))
        faults, never = _faults(a.fault)
        body = {"task_id": a.task, "split": a.split, "mode": a.mode, "agent": a.agent,
                "faults": faults if a.fault else None, "never_commit": never}
        r = cl.post(f"/v1/test/{'runs' if a.cmd == 'run' else 'compare'}", json=body)
        if r.status_code != 200:
            sys.exit(r.text)
        d = r.json()
        if a.cmd == "run":
            print(json.dumps({k: d[k] for k in ("tx_id", "task_id", "mode", "fault_class", "disposition", "decisions")},
                             indent=2))
            print("truth (system level):", {k: d["truth"]["system_level"][k] for k in ("steps", "vsc", "false_success")})
            print("ledger:", cl.get("/v1/ledger/verify").json())
        else:
            for row in d["rows"]:
                print(json.dumps(row))
    elif a.cmd == "ledger":
        from actionproof.integrity.ledger import EvidenceLedger
        led = EvidenceLedger(a.ledger)
        print(json.dumps(led.verify() if a.action == "verify" else led.anchor(a.anchors)))
    elif a.cmd == "contract":
        from actionproof.integrity.contracts import Contract, json_schema
        if a.action == "schema":
            print(json.dumps(json_schema(), indent=2))
        else:
            from actionproof.bench.generator_proc import load_tasks
            from pathlib import Path
            root = Path(__file__).resolve().parents[2]
            t = {x.task_id: x for x in load_tasks(root / f"data/bench/proc/{a.split}.jsonl")}[a.task]
            k = Contract.for_task(t)
            print(json.dumps({"contract": k.model_dump(mode="json"), "content_hash": k.content_hash()}, indent=2))


if __name__ == "__main__":
    main()
