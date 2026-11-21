"""Tool-surface randomisation (PREREGISTRATION_SR1): surfaces are disjoint from the held-out S4 API, and the same
actions under any surface give the same canonical trace, final state and score as under the canonical surface."""
import pytest

from actionproof.bench.generator_proc import DATA, load_tasks
from actionproof.integrity.sim import agents
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.stage3 import scripted
from actionproof.stage3.env import make_env_s3
from actionproof.stage4 import shift, surface as S

TASKS = load_tasks(DATA / "s3_rec_train_half.jsonl")[:60] + load_tasks(DATA / "s3_trainb_half.jsonl")[:60]


class Translate:
    """Wraps a surface env so scripted canonical policies drive it through the surface names only."""

    def __init__(self, env):
        self.env = env

    def call(self, tool, args=None, by="agent"):
        t, a = self.env.to_surface(tool, dict(args or {}))
        obs = self.env.call(t, a, by)
        S_ = self.env.surface
        back = {("status" if k == S_.ack_key else S_.inv_arg.get(k, k)): v for k, v in obs.items()}
        if "status" in back:
            back["status"] = {S_.ok: "ok", S_.err: "error"}.get(back["status"], back["status"])
        if back.get("error") == S_.timeout:
            back["error"] = "timeout"
        return _unrename(back, S_)

    def __getattr__(self, k):
        return getattr(self.env, k)


def _unrename(x, S_):
    if isinstance(x, list):
        return [_unrename(v, S_) for v in x]
    if isinstance(x, dict):
        return {S_.inv_arg.get(k, k): _unrename(v, S_) for k, v in x.items()}
    return x


def test_surfaces_disjoint_from_held_out_api():
    held = set(shift.TOOL_MAP) | set(shift.ARG_MAP.values())
    for s in S.SURFACES:
        assert not (set(s.tool_to_canon) | set(s.arg.values())) & held
    assert len({tuple(sorted(s.tool_to_canon)) for s in S.SURFACES}) == S.K


@pytest.mark.parametrize("k", range(S.K))
def test_same_actions_same_outcome(k):
    for t in TASKS:
        a = make_env_s3(t, "x", seed=0)
        b = S.make_env_surface(t, "x", 0, k)
        for tool, args in t.meta.get("start_prefix") or []:
            a.call(tool, dict(args))
            Translate(b).call(tool, dict(args))
        pol = scripted.oracle if t.meta.get("start_prefix") else agents.oracle
        pol(a, t)
        pol(Translate(b), t)
        assert a.state == b.state
        assert [(e["tool"], e["args"]) for e in a.trace] == [(e["tool"], e["args"]) for e in b.trace]
        assert episode_record_proc(t, a)["score"] == episode_record_proc(t, b)["score"]


def test_canonical_names_rejected_and_acks_renamed():
    t = TASKS[0]
    env = S.make_env_surface(t, "x", 0, 2)
    assert env.call("pay_supplier", {"po_id": "x"})["error"] == "unknown_tool"
    obs = env.call(env.surface.canon_to_tool["get_purchase_order"], {env.surface.arg["po_id"]: t.meta["po_id"]})
    assert "status" not in obs and env.surface.ack_key in obs


def test_choice_shared_within_group_and_balanced():
    ks = [S.choose(f"t{i}", 7) for i in range(4000)]
    assert S.choose("t1", 7) == S.choose("t1", 7)
    assert 0.45 < sum(k is None for k in ks) / len(ks) < 0.55
    assert set(k for k in ks if k is not None) == set(range(S.K))


def test_system_prompt_uses_only_surface_names():
    pytest.importorskip("transformers")
    from actionproof.integrity.sim.prompts import PROC_NEUTRAL
    from actionproof.rollout.episode import system_prompt
    canon = set(S.TOOL_SYN)
    for k in range(S.K):
        env = S.make_env_surface(TASKS[0], "x", 0, k)
        text = system_prompt(env, PROC_NEUTRAL)
        assert all(n in text for n in env.surface.tool_to_canon), k
        leaked = [c for c in canon if f'"{c}"' in text]
        assert not leaked, (k, leaked)


def test_hide_prefix_replays_state_not_conversation():
    pytest.importorskip("transformers")
    import actionproof.stage3.rollout as R
    from actionproof.stage4 import train_variants
    orig = R.start_conversation
    try:
        train_variants.hide_prefix(R)
        t = next(x for x in TASKS if x.meta.get("start_prefix"))
        a, b = make_env_s3(t, "x", seed=0), make_env_s3(t, "x", seed=0)
        conv = R.start_conversation(a, t, "")
        full = orig(b, t, "")
        assert len(conv) == 2 and len(full) > 2 and a.state == b.state
    finally:
        R.start_conversation = orig
