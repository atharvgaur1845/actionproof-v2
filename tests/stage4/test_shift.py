"""Stage 4 shift checks: the API-shifted environment is ProcureSim with a different surface (same final state and score
for the same actions; canonical names rejected; tool schemas show only shifted names), language variants carry every
request field, and the unseen-workflow and fault-combination splits are solvable by the scripted oracle."""
import json

import pytest

pytest.importorskip("transformers")

from actionproof.bench.generator_proc import DATA, load_tasks  # noqa: E402
from actionproof.integrity.sim import agents  # noqa: E402
from actionproof.integrity.sim.runtime import episode_record_proc  # noqa: E402
from actionproof.rollout.episode import system_prompt  # noqa: E402
from actionproof.stage3.env import make_env_s3  # noqa: E402
from actionproof.stage4.shift import ARG_MAP, INV_TOOL, TOOL_MAP, make_env_api  # noqa: E402


class _Translate:
    """Drive the shifted API with canonical calls (for equivalence tests only)."""
    def __init__(self, env):
        self.env, self.trace = env, env.trace

    def call(self, tool, args=None, by="agent"):
        obs = self.env.call(INV_TOOL[tool], {ARG_MAP.get(k, k): v for k, v in (args or {}).items()}, by)
        back = {("status" if k == "result" else k): v for k, v in obs.items()}
        back["status"] = {"ACCEPTED": "ok", "REJECTED": "error"}.get(back.get("status"), back.get("status"))
        inv = {v: k for k, v in ARG_MAP.items()}

        def ren(x):
            if isinstance(x, list):
                return [ren(v) for v in x]
            if isinstance(x, dict):
                return {inv.get(k, k): ren(v) for k, v in x.items()}
            return x
        back = ren(back)
        if back.get("error") == "deadline_exceeded":
            back["error"] = "timeout"
        return back


def test_api_shift_is_procuresim_with_another_surface():
    for t in load_tasks(DATA / "s4_api.jsonl")[:60]:
        a, b = make_env_s3(t, "x", seed=0), make_env_api(t, "x", seed=0)
        agents.oracle(a, t)
        agents.oracle(_Translate(b), t)
        assert a.state == b.state
        assert episode_record_proc(t, a)["score"] == episode_record_proc(t, b)["score"]


def test_api_shift_rejects_canonical_names_and_renders_shifted_schemas():
    t = load_tasks(DATA / "s4_api.jsonl")[0]
    env = make_env_api(t, "x", seed=0)
    assert env.call("pay_supplier", {"po_id": "x", "amount_cents": 1})["error"] == "unknown_tool"
    sp = system_prompt(env, env.adapt_prompt(""))
    for old in TOOL_MAP.values():
        assert f'"{old}"' not in sp, old
    for new in TOOL_MAP:
        assert new in sp


def test_language_variants_carry_the_request():
    en = {t.task_id.split("-")[-1]: t for t in load_tasks(DATA / "s4_lang_en.jsonl")}
    for lang in ("hi", "hinglish", "para"):
        for t in load_tasks(DATA / f"s4_lang_{lang}.jsonl"):
            e = en[t.task_id.split("-")[-1]]
            po = next(p for p in t.initial_state["pos"] if p["po_id"] == t.meta["po_id"])
            assert t.initial_state == e.initial_state and t.meta["faults"] == e.meta["faults"]
            for v in (po["po_id"], str(po["qty"]), str(po["amount_cents"]), po["sku"]):
                assert v in t.prompt, (lang, v)


@pytest.mark.parametrize("split", ["s4_wf4", "s4_combo", "s4_api", "s4_lang_en"])
def test_scripted_calibration(split):
    tasks = load_tasks(DATA / f"{split}.jsonl")
    vsc = {"oracle": 0, "naive": 0}
    for t in tasks:
        for pol in vsc:
            env = make_env_s3(t, "c", seed=0)
            getattr(agents, pol)(env, t)
            vsc[pol] += episode_record_proc(t, env)["score"]["vsc"]
    reachable = sum(1 for t in tasks if not t.meta.get("never_commit"))
    assert vsc["oracle"] >= 0.9 * reachable - 3, (split, vsc, reachable)
    assert vsc["naive"] < vsc["oracle"]
    print(split, {k: round(v / len(tasks), 3) for k, v in vsc.items()})
