"""ActionProof live: run a real Cohere Labs model as a tool-using agent in ProcureSim (simulated tools only)."""
from __future__ import annotations

import os
import threading

import gradio as gr
import spaces

import core
from core import AGENTS, MODELS, REALITIES, REPLAY_URL, SESSION_CAP, run_session, task_choices


# ------------------------------------------------------------------ models (lazy, bfloat16)
def _load_tokenizer(repo: str, token: str | None):
    """The fast tokenizer from tokenizer.json, its special tokens and the chat template.

    transformers 5 writes `tokenizer_class: TokenizersBackend` (and a list-valued `extra_special_tokens`), which the
    4.x AutoTokenizer this Space pins (gradio 4.44.1 needs huggingface_hub < 1.0) cannot resolve. The tokenizer itself
    is the same file either way."""
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from transformers import PreTrainedTokenizerFast
    cfg = json.loads(open(hf_hub_download(repo, "tokenizer_config.json", token=token)).read())
    tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(repo, "tokenizer.json", token=token),
                                  **{k: cfg[k] for k in ("bos_token", "eos_token", "pad_token", "unk_token") if cfg.get(k)})
    try:
        tok.chat_template = open(hf_hub_download(repo, "chat_template.jinja", token=token)).read()
    except EntryNotFoundError:
        ct = cfg.get("chat_template")
        if isinstance(ct, list):  # named templates; with no name given, transformers uses "default"
            ct = {t["name"]: t["template"] for t in ct}.get("default")
        tok.chat_template = ct
    return tok


class HFGenerator:
    """Greedy chat generation, 256 new tokens, tokenizer chat template (the p1_eval settings)."""

    def __init__(self, repo: str) -> None:
        self.repo, self.tok, self.model = repo, None, None
        self._lock = threading.Lock()

    def load_cpu(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM
            token = os.environ.get("HF_TOKEN")  # optional: the models are public
            self.tok = _load_tokenizer(self.repo, token)
            config = AutoConfig.from_pretrained(self.repo, token=token)
            # Checkpoints saved by transformers 5 keep rope_theta inside rope_parameters; 4.x reads only the
            # top-level field and would otherwise fall back silently to its default (10000 instead of 50000).
            rp = getattr(config, "rope_parameters", None)
            if isinstance(rp, dict) and "rope_theta" in rp:
                config.rope_theta = rp["rope_theta"]
            self.model = AutoModelForCausalLM.from_pretrained(self.repo, config=config, torch_dtype=torch.bfloat16,
                                                              token=token)
            self.model.eval()

    def cuda(self) -> None:
        if next(self.model.parameters()).device.type != "cuda":
            self.model.to("cuda")

    def _ids(self, conv: list[dict]):
        text = self.tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False, return_tensors="pt")

    def token_len(self, conv: list[dict]) -> int:
        return int(self._ids(conv)["input_ids"].shape[1])

    def __call__(self, conv: list[dict]) -> str:
        import torch
        enc = self._ids(conv).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                                      max_new_tokens=core.MAX_NEW_TOKENS,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


_GENS = {k: HFGenerator(repo) for k, repo in MODELS.items()}


@spaces.GPU(duration=120)
def _gpu_episode(model_key: str, task_id: str, reality: str):
    gen = _GENS[model_key]
    gen.cuda()
    yield from core.model_events(gen, core.get_task(task_id), reality)


def model_stream(model_key: str, task_id: str, reality: str):
    _GENS[model_key].load_cpu()  # CPU load happens outside the GPU window
    yield from _gpu_episode(model_key, task_id, reality)


# ------------------------------------------------------------------ UI
CSS = """
:root { --accent: #1f5fbf; --ink: #1c2330; --muted: #5b6577; --line: #dde1e8; --soft: #f5f6f9; }
.gradio-container { max-width: 1180px !important; color: var(--ink); }
h1 { font-weight: 650; letter-spacing: -0.01em; margin-bottom: 4px; }
.lede { color: var(--muted); margin: 0 0 14px 0; }
.banner { padding: 8px 12px; border-radius: 6px; font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
.banner.live { background: var(--accent); color: #fff; }
.banner.scripted { background: #fff; color: var(--ink); border: 1.5px solid var(--ink); }
.req, .step, .tbl, .note { border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; margin: 8px 0; background: #fff; }
.req { background: var(--soft); }
.hd, .tt { font-size: 12px; font-weight: 650; color: var(--muted); margin-bottom: 4px; }
.step.report { border-left: 3px solid var(--accent); }
.step.raw pre { margin: 0; white-space: pre-wrap; font-size: 12px; }
.lbl { display: inline-block; width: 42px; color: var(--muted); font-size: 12px; }
code { font-size: 12.5px; background: transparent; word-break: break-word; }
mark.ack { background: #dbe7fa; color: var(--accent); font-weight: 650; padding: 0 3px; border-radius: 3px; }
.note { background: var(--soft); }
.verdict { display: flex; gap: 24px; padding: 10px 12px; border: 1px solid var(--ink); border-radius: 6px; margin: 8px 0; }
.verdict b { font-size: 18px; color: var(--accent); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 3px 8px 3px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
table.kv th { width: 150px; color: var(--muted); font-weight: 600; }
.empty, .fine { color: var(--muted); font-size: 12px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 12px; }
.cmp { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 16px; }
.cmp h4 { margin: 8px 0 0 0; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 760px) { .cols, .cmp, .grid2 { grid-template-columns: 1fr; grid-auto-flow: row; } }
"""

THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate").set(
    button_primary_background_fill="#1f5fbf", button_primary_background_fill_hover="#194f9f",
    button_primary_text_color="#ffffff", body_background_fill="#ffffff", block_background_fill="#ffffff")

INTRO = f"""
# ActionProof live
<p class="lede">The same request and the same "ok" can hide different realities. Pick a task and a reality, run an agent,
and only after the run see what the databases actually hold. All tools are simulated; nothing touches a real system.
Deterministic replays of recorded runs: <a href="{REPLAY_URL}" target="_blank">{REPLAY_URL}</a></p>
"""


def build_ui(stream=model_stream) -> gr.Blocks:
    tasks = task_choices()
    with gr.Blocks(theme=THEME, css=CSS, title="ActionProof live") as demo:
        gr.HTML(INTRO)
        state = gr.State({"runs": 0})
        with gr.Row():
            task = gr.Dropdown(list(tasks), value=list(tasks)[0], label="Task (curated test tasks with a pay step)")
            reality = gr.Dropdown(list(REALITIES), value="Reality A",
                                  label="Reality (what the payment system really does; revealed after the run)")
            agent = gr.Dropdown([(v, k) for k, v in AGENTS.items()], value="scripted_naive", label="Agent")
        with gr.Row():
            run = gr.Button("Run", variant="primary")
            status = gr.Markdown(f"Live model runs are limited to {SESSION_CAP} per session and run one at a time. "
                                 "Scripted runs need no GPU.")
        with gr.Row():
            with gr.Column():
                ban_l = gr.HTML()
                left = gr.HTML()
            with gr.Column():
                ban_r = gr.HTML()
                right = gr.HTML()
        reveal = gr.HTML()

        def go(agent_key, task_label, reality_key, st):
            try:
                for ui in run_session(agent_key, tasks[task_label], reality_key, st, stream):
                    yield (ui["banner_l"], ui["left"], ui["banner_r"], ui["right"], ui["reveal"], ui["status"], st)
            except Exception as e:  # GPU quota, model download failure, etc.
                yield ("", "", "", "", "", f"Run failed: {type(e).__name__}: {e}", st)

        run.click(go, [agent, task, reality, state], [ban_l, left, ban_r, right, reveal, status, state],
                  concurrency_limit=1, concurrency_id="run")
    return demo


demo = build_ui()
demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
