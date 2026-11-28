"""Teachers: propose a continuation from an agent's failure prefix.

A teacher sees only what the agent saw: the conversation so far (request, tool results) and the env through
agent-visible tools. It never reads the fault plan, the attempts table or the hidden state. Whatever it proposes
is replayed and scored by the verifier afterwards (`replay.verify_continuation`); a teacher's own belief that it
succeeded counts for nothing.

Backends:
- `B4Teacher`: TXN's transaction-aware reconciliation policy B4 (`actionproof/txn/policies.py`), adapted to start
  from an arbitrary prefix: it reads before it writes, so steps the agent already completed are not redone, and it
  chooses idempotency keys from the visible history.
- `LLMTeacher`: any OpenAI-compatible chat endpoint (for example vLLM serving Aya Expanse 32B on the cluster).
  Not run in this branch; it is exercised in tests with a fake client.

Register more backends in `TEACHERS`.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from actionproof.integrity.sim.agents import _args, _observed
from actionproof.integrity.sim.env import STEP_READ, STEP_TOOL
from actionproof.selfheal.envs import EnvSpec
from actionproof.selfheal.replay import Replayed, act_text, execute_turn, parser, tool_text

MAX_WRITES = 3  # policies.MAX_WRITES


class OutOfTurns(Exception):
    pass


@dataclass
class Context:
    """The live episode a teacher continues. `call` runs one tool as one assistant turn and records both messages."""
    spec: EnvSpec
    rep: Replayed
    actions: list[dict[str, str]] = field(default_factory=list)  # assistant/user messages after the cut

    @property
    def task(self) -> Any:
        return self.rep.task

    @property
    def env(self) -> Any:
        return self.rep.env

    @property
    def conv(self) -> list[dict[str, str]]:
        return self.rep.conv + self.actions

    @property
    def turns_left(self) -> int:
        return self.rep.turns_left - len(self.actions) // 2

    def history(self) -> list[dict[str, Any]]:
        """Agent-visible calls so far (tool, args, observation); colleague writes are not visible."""
        return [{"tool": s["tool"], "args": s["args"], "obs": s["obs"]} for s in self.env.trace
                if s.get("by", "agent") == "agent"]

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.env.done:
            raise OutOfTurns("episode finished")
        if self.turns_left <= 0 or (self.turns_left == 1 and tool != "submit_report"):
            raise OutOfTurns(tool)
        text = act_text(tool, args)
        n = len(self.env.trace)
        res, _, _ = execute_turn(self.env, text)
        self.actions += [{"role": "assistant", "content": text}, {"role": "user", "content": "\n\n".join(res)}]
        mine = [s for s in self.env.trace[n:] if s.get("by", "agent") == "agent"]  # a colleague write may land first
        return mine[-1]["obs"] if mine else {}

    def raw_turn(self, text: str) -> bool:
        """Run a free-text assistant turn (LLM teachers). Returns False when it had no parseable call."""
        if self.turns_left <= 0 or self.env.done:
            raise OutOfTurns("raw")
        res, _, had = execute_turn(self.env, text)
        self.actions.append({"role": "assistant", "content": text})
        if had:
            self.actions.append({"role": "user", "content": "\n\n".join(res)})
        return had


class Teacher(Protocol):
    name: str

    def continue_from(self, ctx: Context) -> None: ...


# ---------------------------------------------------------------------- B4 reconciliation teacher
def _ok(obs: dict[str, Any]) -> bool:
    return obs.get("status") == "ok"


def _live(obs: dict[str, Any]) -> list[dict[str, Any]]:
    refunded = {p["refund_of"] for p in obs["payments"] if p["kind"] == "refund"}
    return [p for p in obs["payments"] if p["kind"] == "payment" and p["payment_id"] not in refunded]


def _received(obs: dict[str, Any]) -> int:
    return sum(r["qty"] if r["kind"] == "receipt" else -r["qty"] for r in obs["receipts"])


ID_FIELD = {"order": "order_ref", "pay": "payment_id", "post": "entry_id"}
ROWS = {"order": "orders", "pay": "payments", "post": "entries"}


class B4Teacher:
    """B4 (read-back, consistent reads, same-key retry, decomposed repair, escalation when unreadable), started from
    a prefix. Differences from `policies.make_b4`, all forced by starting mid-episode:
    - each step starts with a read, not a write;
    - an absent effect is retried with the last key this episode used for the step, unless the visible history
      shows that key's effect was already used up (its acknowledged id is visible but no longer live); otherwise a
      key this episode has not used;
    - without consistent reads (P1 env) an absent or partial reading is confirmed by a second read before writing;
    - a duplicate payment keeps one full live payment (the earliest) and refunds the others;
    - when retries are exhausted: escalate on TXN (B4), report "failed" on P1 (the P1 oracle).
    """

    name = "b4"

    def __init__(self, max_writes: int = MAX_WRITES) -> None:
        self.max_writes = max_writes

    # -- keys
    @staticmethod
    def _used_keys(ctx: Context) -> set[str]:
        return {h["args"].get("idempotency_key") for h in ctx.history() if isinstance(h["args"], dict)} - {None}

    def _fresh(self, ctx: Context, step: str, pid: str) -> str:
        used = self._used_keys(ctx)
        base = f"{step}-{pid}"
        if base not in used:
            return base
        n = 1
        while f"{base}-sh{n}" in used:
            n += 1
        return f"{base}-sh{n}"

    def _retry_key(self, ctx: Context, step: str, pid: str, read: dict[str, Any]) -> str:
        tool = STEP_TOOL[step]
        mine = [h for h in ctx.history() if h["tool"] == tool and isinstance(h["args"], dict)
                and h["args"].get("po_id") == pid and h["args"].get("idempotency_key")]
        if not mine:
            return self._fresh(ctx, step, pid)
        key = mine[-1]["args"]["idempotency_key"]
        visible = {row.get(ID_FIELD[step]) for row in read.get(ROWS[step], [])}
        acked = {h["obs"].get(ID_FIELD[step]) for h in mine if h["args"]["idempotency_key"] == key and _ok(h["obs"])}
        if acked & visible:  # the key's effect exists but is not live (refunded or superseded): the key is used up
            return self._fresh(ctx, step, pid)
        return key

    # -- reads and writes
    def _read(self, ctx: Context, step: str, pid: str) -> dict[str, Any]:
        a: dict[str, Any] = {"po_id": pid}
        if ctx.spec.consistent_reads:
            a["consistent"] = True
        return ctx.call(STEP_READ[step], a)

    def _escalate(self, ctx: Context, pid: str, reason: str) -> str:
        ctx.call("escalate_to_human", {"po_id": pid, "reason": reason})
        ctx.call("submit_report", {"status": "escalated", "summary": reason})
        return "escalated"

    def _exhausted(self, ctx: Context, step: str, pid: str, writes: int) -> str:
        if ctx.spec.name.startswith("txn"):
            return self._escalate(ctx, pid, f"{step} for {pid} never took effect after {writes} attempts")
        return "failed"

    def _step(self, ctx: Context, step: str, po: dict[str, Any]) -> str:
        pid, writes = po["po_id"], 0
        for _ in range(8):
            obs = self._read(ctx, step, pid)
            if not _ok(obs):
                if ctx.spec.consistent_reads:  # TXN unresolvable regime: the record cannot be read
                    return self._escalate(ctx, pid, f"cannot confirm {step} for {pid}: its records are unavailable")
                return "failed"
            seen = _observed(step, obs, po)
            if seen in ("absent", "partial") and not ctx.spec.consistent_reads:
                again = self._read(ctx, step, pid)  # a replica may lag; trust two agreeing reads
                if not _ok(again):
                    return "failed"
                obs, seen = again, _observed(step, again, po)
            if seen == "held":
                return "held"
            if seen == "absent":
                if writes >= self.max_writes:
                    return self._exhausted(ctx, step, pid, writes)
                key = None if step == "receive" else self._retry_key(ctx, step, pid, obs)
                ctx.call(STEP_TOOL[step], _args(step, po, key))
                writes += 1
                continue
            if seen == "partial" and step == "pay":
                ctx.call("refund_payment", {"payment_id": _live(obs)[-1]["payment_id"]})
                ctx.call(STEP_TOOL[step], _args(step, po, self._fresh(ctx, step, pid)))
                writes += 1
                continue
            if seen == "partial" and step == "receive":
                a = _args(step, po, None)
                a["qty"] = po["qty"] - _received(obs)
                ctx.call(STEP_TOOL[step], a)
                writes += 1
                continue
            if seen == "other" and step == "pay" and len(_live(obs)) > 1:
                live = _live(obs)
                keep = next((p for p in live if p["amount_cents"] == po["amount_cents"]), live[0])
                for p in live:
                    if p is not keep:
                        ctx.call("refund_payment", {"payment_id": p["payment_id"]})
                continue
            return "failed"  # a duplicate order or journal entry, or an over-receipt: no agent tool repairs it
        return "failed"

    def continue_from(self, ctx: Context) -> None:
        from actionproof.integrity.sim.agents import _po
        po = _po(ctx.task)
        done: list[str] = []
        try:
            for step in ctx.task.meta["requested"]:
                r = self._step(ctx, step, po)
                if r == "escalated":
                    return
                if r == "failed":
                    ctx.call("submit_report", {"status": "failed", "summary":
                             f"could not complete {step} for {po['po_id']}; done: {', '.join(done) or 'nothing'}"})
                    return
                done.append(step)
            ctx.call("submit_report", {"status": "completed", "summary":
                     f"{', '.join(done)} done for {po['po_id']}; each step confirmed in its system of record"})
        except OutOfTurns:
            return


# ---------------------------------------------------------------------- LLM teacher (OpenAI-compatible endpoint)
TEACHER_NOTE = ("\n\nYou are taking over a task that another operator started. Their earlier actions and the tool "
                "results are above; some of their actions may have been wrong or left the records inconsistent. "
                "Before any write, check the system of record with the read tools. Never repeat a write that already "
                "took effect. Repair partial or duplicate effects with the available tools. Report \"completed\" only "
                "when every requested step is confirmed in its system of record.")


def http_client(base: str, model: str, key: str | None = None, temperature: float = 0.0, max_tokens: int = 256,
                timeout: float = 120.0) -> Callable[[list[dict[str, str]]], str]:
    def complete(messages: list[dict[str, str]]) -> str:
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {key}"} if key else {})})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())["choices"][0]["message"]["content"] or ""
            except OSError:
                time.sleep(2 ** attempt)
        return ""
    return complete


class LLMTeacher:
    """Continues the conversation turn by turn with a chat model. The model's first parsed call of each turn is
    executed and stored in canonical form (`act_text`), so SFT targets have the training format and no free text;
    a turn without a parseable call ends the continuation (it will then fail verification)."""

    name = "llm"

    def __init__(self, complete: Callable[[list[dict[str, str]]], str], note: str = TEACHER_NOTE) -> None:
        self.complete, self.note = complete, note

    def continue_from(self, ctx: Context) -> None:
        parse = parser()
        try:
            while not ctx.env.done and ctx.turns_left > 0:
                msgs = [dict(m) for m in ctx.conv]
                msgs[0]["content"] += self.note
                calls = parse(self.complete(msgs)) or []
                if not calls:
                    return
                c = calls[0]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    return
                if not isinstance(args, dict):
                    return
                ctx.call(c["function"]["name"], args)
        except OutOfTurns:
            return


TEACHERS: dict[str, Callable[..., Teacher]] = {"b4": B4Teacher, "llm": LLMTeacher}


__all__ = ["Context", "Teacher", "B4Teacher", "LLMTeacher", "http_client", "TEACHERS", "OutOfTurns", "tool_text"]
