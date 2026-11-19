"""Thin client for agents running in their own process. The agent holds only the gateway URL; service access,
contracts and evidence stay inside the gateway process.

    from actionproof.integrity.sdk import Gateway
    gw = Gateway("http://127.0.0.1:8765")
    with gw.transaction(task_id="proc_dev-0007", mode="enforce") as tx:
        tools = tx.tools()                      # callables with the services' names; every call is mediated
        report = my_agent(tools, tx.prompt)     # existing agent code, unchanged
        result = tx.finalize()                  # the agent's claim is recorded; the gateway decides
    print(result["status"], result["claim"], result["exposure_cents"])
"""
from __future__ import annotations

from typing import Any, Callable

import httpx


class Transaction:
    def __init__(self, client: httpx.Client, tx_id: str) -> None:
        self.client, self.tx_id = client, tx_id
        meta = self.client.get(f"/v1/transactions/{tx_id}/tools").raise_for_status().json()
        self.prompt, self.system_prompt, self.schemas = meta["user_prompt"], meta["system_prompt"], meta["tools"]

    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.client.post(f"/v1/transactions/{self.tx_id}/calls",
                                json={"tool": tool, "args": args or {}}).raise_for_status().json()

    def tools(self) -> list[Callable[..., dict[str, Any]]]:
        def make(name: str, doc: str) -> Callable[..., dict[str, Any]]:
            def tool(**kwargs: Any) -> dict[str, Any]:
                return self.call(name, kwargs)
            tool.__name__ = name
            tool.__doc__ = doc
            return tool
        return [make(s.get("name") or s.get("function", {}).get("name"),
                     s.get("description") or s.get("function", {}).get("description", "")) for s in self.schemas]

    def finalize(self) -> dict[str, Any]:
        return self.client.post(f"/v1/transactions/{self.tx_id}/finalize").raise_for_status().json()

    def detail(self) -> dict[str, Any]:
        return self.client.get(f"/v1/transactions/{self.tx_id}").raise_for_status().json()

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class Gateway:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 30.0) -> None:
        self.client = httpx.Client(base_url=url, timeout=timeout)

    def transaction(self, task_id: str, split: str = "proc_dev", mode: str = "enforce",
                    faults: list[dict[str, Any]] | None = None, never_commit: str | None = None) -> Transaction:
        r = self.client.post("/v1/transactions", json={"task_id": task_id, "split": split, "mode": mode,
                                                      "faults": faults, "never_commit": never_commit})
        return Transaction(self.client, r.raise_for_status().json()["tx_id"])

    def verify_ledger(self) -> dict[str, Any]:
        return self.client.get("/v1/ledger/verify").raise_for_status().json()
