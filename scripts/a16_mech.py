"""A16: representation and causal action selection at the post-evidence decision point (PREREGISTRATION_A15).

Decision points come from A15's forks (`scripts/a15_eval.py`): the forced `list_payments` result shows the stored
payment, q = 0, 1/4, 1/2, 3/4 or 1 of the purchase-order amount. For each model:
  margin   M = log P(refund or escalate) - log P(post), teacher-forced over the first tool name
           ('[{"tool_name": "' + tool + '",'), exact (no sampling).
  probes   last-prompt-token residual stream at every layer; partial (q < 1) vs complete (q = 1); a probe fitted on
           the warm start is applied frozen to later checkpoints, and a probe is refitted per checkpoint.
  patch    residual stream of a complete-branch donor written into the partial branch (and back) at one layer, at
           the final prompt token or at the token span where the evidence differs; with self, random same-norm and
           wrong-task controls.
  ablate   the difference-of-means direction (partial - complete, dev split) projected out at one layer.
Hidden-state index L (0 = embeddings) is the output of decoder layer L-1; hooks are placed accordingly and checked
against `output_hidden_states` before any measurement (`check_hooks`). The last index (after the final norm) is never
patched.

    python scripts/a16_mech.py trajectory --models warm=runs/rft_proc/merged,rv_s0_200=runs/a12_proc_rv_s0/checkpoint-200
    python scripts/a16_mech.py sweep --models rv_s0=runs/a12_proc_rv_s0/final        # dev only; writes the layer choice
    python scripts/a16_mech.py causal --models rv_s0=...,ra_s0=...                   # test, at the chosen layer
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a15_eval", ROOT / "scripts/a15_eval.py")
a15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a15)

BASE = "runs/rft_proc/merged"
STEM, CLOSE = '[{"tool_name": "', '",'
CANDS = ("refund_payment", "escalate_to_human", "post_journal_entry", "pay_supplier", "list_payments", "submit_report")
RECOVER, PROCEED = ("refund_payment", "escalate_to_human"), "post_journal_entry"
PARTIAL, COMPLETE = ("Q1", "Q2", "Q3"), "Q4"
N_RANDOM, N_CONTROL_PAIRS, MIN_GAP = 20, 50, 1.0
OUT = ROOT / "results/a16"


def lse(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def margin(lp: dict[str, float]) -> float:
    return lse([lp[c] for c in RECOVER]) - lp[PROCEED]


# ------------------------------------------------------------------ examples
def examples(split: str, n: int | None = None) -> list[dict[str, Any]]:
    from actionproof.bench.generator_proc import load_tasks
    out = []
    for t in load_tasks(ROOT / f"data/bench/proc/{split}.jsonl")[:n]:
        for b in a15.BRANCHES:
            _, conv, _ = a15.cef.build(t, b)
            out.append({"task_id": t.task_id, "branch": b, "q": a15.QUARTERS[b] / 4, "conv": conv})
    return out


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ROOT / BASE)
        p = ROOT / path
        if (p / "adapter_config.json").exists():
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(ROOT / BASE, dtype=torch.bfloat16)
            self.model = PeftModel.from_pretrained(base, p).merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16)
        self.model = self.model.cuda().eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.close_id = self.tok(CLOSE, add_special_tokens=False).input_ids
        assert len(self.close_id) == 1, self.close_id

    def prompt_ids(self, conv: list[dict[str, str]]) -> list[int]:
        text = self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False).input_ids

    def seqs(self, conv: list[dict[str, str]], cands=CANDS) -> tuple[list[int], list[list[int]], int]:
        text = self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        p = self.tok(text, add_special_tokens=False).input_ids
        ss = [self.tok(text + STEM + c + CLOSE, add_special_tokens=False).input_ids for c in cands]
        lcp = 0
        while all(len(s) > lcp for s in ss) and len({s[lcp] for s in ss}) == 1:
            lcp += 1
        assert lcp >= len(p) and ss[0][:len(p)] == p, "prompt is not a token prefix of the scored sequences"
        return p, ss, lcp

    def _hook(self, L: int, fn):
        """Hook producing hidden-state index L (output of decoder layer L-1); fn(h) edits h in place."""
        assert 1 <= L < self.n_layers, L

        def h(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            fn(t)
            return out
        return self.layers[L - 1].register_forward_hook(h)

    def run(self, conv, cands=CANDS, edit: tuple[int, Any] | None = None, hidden: bool = False):
        """Log P of each candidate's tool name (+ CLOSE); optionally edit the residual stream at index L during the
        shared-prefix pass, and return last-prompt-token hidden states at every index."""
        torch = self.torch
        p, ss, lcp = self.seqs(conv, cands)
        handle = self._hook(edit[0], edit[1]) if edit else None
        try:
            with torch.no_grad():
                out = self.model(torch.tensor([ss[0][:lcp]]).cuda(), use_cache=True, output_hidden_states=hidden)
        finally:
            if handle:
                handle.remove()
        first = torch.log_softmax(out.logits[0, -1].float(), -1)
        lps = {}
        with torch.no_grad():
            for c, s in zip(cands, ss):
                v = first[s[lcp]].item()
                if len(s) > lcp + 1:
                    o = self.model(torch.tensor([s[lcp:-1]]).cuda(), past_key_values=copy.deepcopy(out.past_key_values),
                                   use_cache=True)
                    lp = torch.log_softmax(o.logits[0].float(), -1)
                    v += sum(lp[j, s[lcp + 1 + j]].item() for j in range(len(s) - lcp - 1))
                lps[c] = v
        hs = None
        if hidden:
            hs = torch.stack([h[0, len(p) - 1].float() for h in out.hidden_states]).cpu().numpy()
        return lps, hs, len(p)

    def states(self, conv, L: int, positions: list[int]):
        torch = self.torch
        p = self.prompt_ids(conv)
        with torch.no_grad():
            out = self.model(torch.tensor([p]).cuda(), output_hidden_states=True)
        return out.hidden_states[L][0, positions].clone()


def check_hooks(m: Model, conv) -> None:
    """Captured layer outputs must equal output_hidden_states at the same index, and a self-patch must not move M."""
    torch = m.torch
    p = m.prompt_ids(conv)
    ids = torch.tensor([p]).cuda()
    with torch.no_grad():
        ref = m.model(ids, output_hidden_states=True).hidden_states
    for L in (1, m.n_layers // 2, m.n_layers - 1):
        got = {}
        hd = m._hook(L, lambda h: got.setdefault("h", h.clone()))
        with torch.no_grad():
            m.model(ids)
        hd.remove()
        assert torch.allclose(got["h"][0], ref[L][0], atol=1e-3), f"hook index mismatch at L={L}"
    L, pos = m.n_layers // 2, len(p) - 1
    own = ref[L][0, pos].clone()
    base = margin(m.run(conv, (RECOVER + (PROCEED,)))[0])

    def put(h):
        h[0, pos] = own
    patched = margin(m.run(conv, (RECOVER + (PROCEED,)), edit=(L, put))[0])
    assert abs(patched - base) < 1e-3, ("self-patch moved M", base, patched)


# ------------------------------------------------------------------ probes
def probe_eval(Xtr, ytr, Xte, yte, frozen=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = frozen or make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.decision_function(Xte))), clf


# ------------------------------------------------------------------ commands
def cmd_trajectory(models: list[tuple[str, str]], n_test: int) -> None:
    import numpy as np
    dev, test = examples("proc_mech_dev"), examples("proc_mech", n_test)
    frozen: dict[int, Any] = {}
    for tag, path in models:
        out = OUT / f"trajectory__{tag}.json"
        if out.exists():
            continue
        t0 = time.time()
        m = Model(path)
        check_hooks(m, dev[1]["conv"])
        res = {"tag": tag, "path": path, "n_layers": m.n_layers, "examples": {}}
        H = {}
        for split, exs in (("dev", dev), ("test", test)):
            rows, hs = [], []
            for e in exs:
                lps, h, _ = m.run(e["conv"], hidden=True)
                rows.append({"task_id": e["task_id"], "branch": e["branch"], "lp": lps, "M": margin(lps)})
                hs.append(h)
            res["examples"][split] = rows
            H[split] = (np.stack(hs), [e["branch"] for e in exs])
        sel = {s: [i for i, b in enumerate(H[s][1]) if b in PARTIAL + (COMPLETE,)] for s in H}
        y = {s: np.array([H[s][1][i] in PARTIAL for i in sel[s]], int) for s in H}
        res["probe"] = {"refit": [], "frozen_warm": []}
        for L in range(1, m.n_layers):
            Xd, Xt = H["dev"][0][sel["dev"], L], H["test"][0][sel["test"], L]
            auc, clf = probe_eval(Xd, y["dev"], Xt, y["test"])
            res["probe"]["refit"].append(auc)
            if tag == models[0][0]:
                frozen[L] = clf
            res["probe"]["frozen_warm"].append(probe_eval(None, None, Xt, y["test"], frozen[L])[0] if L in frozen else None)
        res["seconds"] = round(time.time() - t0)
        out.write_text(json.dumps(res))
        print(tag, "done", res["seconds"], "s", flush=True)
        del m
        __import__("torch").cuda.empty_cache()


def pairs(exs: list[dict]) -> list[tuple[dict, dict]]:
    by = {(e["task_id"], e["branch"]): e for e in exs}
    return [(by[(t, b)], by[(t, COMPLETE)]) for t in dict.fromkeys(e["task_id"] for e in exs) for b in PARTIAL]


def diff_span(m: Model, a: dict, b: dict) -> list[int] | None:
    pa, pb = m.prompt_ids(a["conv"]), m.prompt_ids(b["conv"])
    if len(pa) != len(pb):
        return None
    return [i for i, (x, y) in enumerate(zip(pa, pb)) if x != y]


def patch_effect(m: Model, base: dict, donor: dict, L: int, pos: list[int], m_base: float, m_donor: float,
                 vec=None) -> float | None:
    if abs(m_donor - m_base) < MIN_GAP:
        return None
    v = m.states(donor["conv"], L, pos) if vec is None else vec

    def put(h):
        h[0, pos] = v.to(h.dtype)
    m_p = margin(m.run(base["conv"], RECOVER + (PROCEED,), edit=(L, put))[0])
    return (m_p - m_base) / (m_donor - m_base)


def cmd_sweep(models: list[tuple[str, str]], n_pairs: int) -> None:
    """Dev only. Per layer, mean normalised effect of the final-token complete->partial patch; the registered rule
    picks the layer with the largest mean effect for the first model (R_V seed 0)."""
    exs = examples("proc_mech_dev")
    res = {}
    for tag, path in models:
        m = Model(path)
        check_hooks(m, exs[1]["conv"])
        M = {(e["task_id"], e["branch"]): margin(m.run(e["conv"], RECOVER + (PROCEED,))[0]) for e in exs}
        ps = pairs(exs)[:n_pairs]
        per = {}
        for L in range(1, m.n_layers):
            effs = []
            for a, b in ps:
                pos = [len(m.prompt_ids(a["conv"])) - 1]
                e = patch_effect(m, a, b, L, pos, M[(a["task_id"], a["branch"])], M[(b["task_id"], b["branch"])])
                if e is not None:
                    effs.append(e)
            per[L] = {"mean": sum(effs) / len(effs) if effs else None, "n": len(effs)}
        res[tag] = {"per_layer": per, "n_pairs": len(ps)}
        del m
        __import__("torch").cuda.empty_cache()
    first = res[models[0][0]]["per_layer"]
    ok = {L: v["mean"] for L, v in first.items() if v["mean"] is not None}
    res["chosen_layer"] = max(ok, key=ok.get) if ok else None
    res["rule"] = "argmax over layers of the mean normalised final-token patch effect, first model, dev pairs"
    (OUT / "sweep_dev.json").write_text(json.dumps(res, indent=1))
    print("chosen layer", res["chosen_layer"])


def cmd_causal(models: list[tuple[str, str]], n_test: int) -> None:
    import numpy as np
    L = json.loads((OUT / "sweep_dev.json").read_text())["chosen_layer"]
    assert L is not None, "no layer chosen on dev"
    dev, test = examples("proc_mech_dev"), examples("proc_mech", n_test)
    rng = random.Random(0)
    for tag, path in models:
        out = OUT / f"causal__{tag}.json"
        if out.exists():
            continue
        m = Model(path)
        check_hooks(m, dev[1]["conv"])
        torch = m.torch
        M = {(e["task_id"], e["branch"]): margin(m.run(e["conv"], RECOVER + (PROCEED,))[0]) for e in test}
        key = lambda e: (e["task_id"], e["branch"])
        res = {"tag": tag, "layer": L, "patch": [], "controls": [], "ablation": {}}
        ps = pairs(test)
        for a, b in ps:
            fin_a, fin_b = [len(m.prompt_ids(a["conv"])) - 1], [len(m.prompt_ids(b["conv"])) - 1]
            span = diff_span(m, a, b)
            row = {"task_id": a["task_id"], "branch": a["branch"], "M_partial": M[key(a)], "M_complete": M[key(b)],
                   "aligned": span is not None,
                   "final_c2p": patch_effect(m, a, b, L, fin_a, M[key(a)], M[key(b)]),
                   "final_p2c": patch_effect(m, b, a, L, fin_b, M[key(b)], M[key(a)])}
            if span:
                row["span_c2p"] = patch_effect(m, a, b, L, span, M[key(a)], M[key(b)])
                row["span_p2c"] = patch_effect(m, b, a, L, span, M[key(b)], M[key(a)])
            res["patch"].append(row)
            if len(res["controls"]) < N_CONTROL_PAIRS and abs(M[key(b)] - M[key(a)]) >= MIN_GAP:
                own = m.states(a["conv"], L, fin_a)
                don = m.states(b["conv"], L, fin_b)
                norm = (don - own).norm()
                rand = []
                for _ in range(N_RANDOM):
                    d = torch.randn_like(own)
                    rand.append(patch_effect(m, a, b, L, fin_a, M[key(a)], M[key(b)], vec=own + d / d.norm() * norm))
                other = rng.choice([p for p in ps if p[0]["task_id"] != a["task_id"]])[1]
                res["controls"].append({
                    "task_id": a["task_id"], "branch": a["branch"],
                    "self": patch_effect(m, a, b, L, fin_a, M[key(a)], M[key(b)], vec=own),
                    "random": rand,
                    "wrong_task": patch_effect(m, a, b, L, fin_a, M[key(a)], M[key(b)],
                                               vec=m.states(other["conv"], L, [len(m.prompt_ids(other["conv"])) - 1]))})
        # direction ablation: difference of means on dev at layer L, final token, this model
        dv = {}
        for e in dev:
            if e["branch"] in PARTIAL + (COMPLETE,):
                dv.setdefault(e["branch"] in PARTIAL, []).append(
                    m.states(e["conv"], L, [len(m.prompt_ids(e["conv"])) - 1])[0].float())
        u = torch.stack(dv[True]).mean(0) - torch.stack(dv[False]).mean(0)
        u = u / u.norm()

        def ablate(direction, start):
            """Project the direction out from the final prompt token onwards (the token the direction was fitted
            at, and the shared tool-call stem that follows it in the prefix pass)."""
            def fn(h):
                d = direction.to(h.device)
                x = h[0, start:].float()
                h[0, start:] = (x - (x @ d)[:, None] * d[None, :]).to(h.dtype)
            return fn
        dirs = [("diff_means", u)] + [(f"random{j}", (lambda r: r / r.norm())(torch.randn_like(u))) for j in range(N_RANDOM)]
        for name, d in dirs:
            exs = test if name == "diff_means" else test[: 5 * 20]
            deltas = {b: [] for b in a15.BRANCHES}
            for e in exs:
                lp = m.run(e["conv"], CANDS, edit=(L, ablate(d, len(m.prompt_ids(e["conv"])) - 1)))[0]
                deltas[e["branch"]].append(margin(lp) - M[key(e)])
            res["ablation"][name] = {b: (float(np.mean(v)) if v else None) for b, v in deltas.items()}
        out.write_text(json.dumps(res))
        print(tag, "causal done", flush=True)
        del m
        torch.cuda.empty_cache()


def parse_models(s: str) -> list[tuple[str, str]]:
    return [tuple(x.split("=", 1)) for x in s.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trajectory", "sweep", "causal"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--out-dir", default="results/a16")
    a = ap.parse_args()
    global OUT
    OUT = ROOT / a.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    ms = parse_models(a.models)
    if a.cmd == "trajectory":
        cmd_trajectory(ms, a.n_test or 100)
    elif a.cmd == "sweep":
        cmd_sweep(ms, a.n_pairs)
    else:
        cmd_causal(ms, a.n_test)


if __name__ == "__main__":
    main()
