"""Record real model runs for the Space's labelled fallback (run on a GPU machine, not in the Space).

Every curated task x reality, for the base model and the trained agent, through the Space's own episode loop
(`core.model_events`: the same prompt, tools, parser, 16 turns, 256 new tokens, greedy). The Space replays these only
under a "RECORDED MODEL RUN" banner, never as a live run.

    python space_live/record_runs.py --base <dir or repo> --trained <dir or repo> --trained-revision <sha> \
        --hardware "A100 80GB (sharanga)" --out space_live/data/recorded.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import core  # noqa: E402


class Gen:
    """The Space's HFGenerator settings: chat template, no added special tokens, greedy, MAX_NEW_TOKENS."""

    def __init__(self, path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()

    def _ids(self, conv):
        text = self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False, return_tensors="pt")

    def token_len(self, conv) -> int:
        return int(self._ids(conv)["input_ids"].shape[1])

    def __call__(self, conv) -> str:
        import torch
        enc = {k: v.to("cuda") for k, v in self._ids(conv).items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=core.MAX_NEW_TOKENS,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--base-revision", default=None)
    ap.add_argument("--trained-revision", default=None)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--out", default=str(HERE / "data/recorded.json"))
    a = ap.parse_args()
    import transformers
    runs = {}
    for key, path, rev in (("base", a.base, a.base_revision), ("trained", a.trained, a.trained_revision)):
        gen = Gen(path)
        for label, tid in core.task_choices().items():
            for reality in core.REALITIES:
                evs = [ev for ev in core.model_events(gen, core.get_task(tid), reality)]
                runs[f"{key}|{tid}|{reality}"] = evs
                end = evs[-1]
                print(key, tid, reality, end["ended_by"], "vsc", end["rec"]["score"]["vsc"], flush=True)
        del gen
    meta = {"recorded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ"), "hardware": a.hardware,
            "models": {"base": {"repo": core.BASE_REPO, "path": a.base, "revision": a.base_revision},
                       "trained": {"repo": core.TRAINED_REPO, "path": a.trained, "revision": a.trained_revision}},
            "decoding": f"greedy, max {core.MAX_TURNS} turns, {core.MAX_NEW_TOKENS} new tokens per turn",
            "transformers": transformers.__version__,
            "note": "Real model outputs recorded through the Space's episode loop; replayed only under a RECORDED banner."}
    Path(a.out).write_text(json.dumps({"meta": meta, "runs": runs}, default=str))
    print("wrote", a.out, len(runs), "runs")


if __name__ == "__main__":
    main()
