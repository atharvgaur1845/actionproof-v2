"""ActionProof live: run a real Cohere Labs model as a tool-using agent in ProcureSim (simulated tools only)."""
from __future__ import annotations

import json
import os
import tempfile
import threading

import huggingface_hub

if not hasattr(huggingface_hub, "HfFolder"):  # gradio 4.44.1 imports HfFolder, which huggingface_hub 1.x removed
    class HfFolder:  # minimal shim: gradio's oauth module only reads or clears the stored token
        get_token = staticmethod(huggingface_hub.get_token)

        @staticmethod
        def save_token(token: str) -> None:
            huggingface_hub.login(token=token, add_to_git_credential=False)

        @staticmethod
        def delete_token() -> None:
            huggingface_hub.logout()
    huggingface_hub.HfFolder = HfFolder

import gradio as gr  # noqa: E402
import spaces  # noqa: E402

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
    tok.model_input_names = ["input_ids", "attention_mask"]  # the bare fast tokenizer also emits token_type_ids
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
            token = os.environ.get("HF_TOKEN")  # needed for the gated base model; the trained agent is public
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
try:
    import transformers as _tf
    print(f"versions: gradio {gr.__version__}, huggingface_hub {huggingface_hub.__version__}, transformers {_tf.__version__}",
          flush=True)
except Exception as _e:  # diagnostics only
    print("versions: unavailable", _e, flush=True)


@spaces.GPU(duration=120)
def _gpu_episode(model_key: str, task_id: str, reality: str):
    gen = _GENS[core.MODEL_OF.get(model_key, model_key)]
    gen.cuda()
    yield from core.model_events(gen, core.get_task(task_id), reality, gate=model_key in core.GATED)


def model_stream(model_key: str, task_id: str, reality: str):
    _GENS[core.MODEL_OF.get(model_key, model_key)].load_cpu()  # CPU load happens outside the GPU window
    yield from _gpu_episode(model_key, task_id, reality)


# ------------------------------------------------------------------ UI
CSS = """
@import url("https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=IBM+Plex+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap");
:root { --paper:#F1EFE7; --ink:#171815; --min:#E3E0D6; --ver:#F05236; --ver-t:#C8381F; --chr:#D5F36A; --gr:#777A72;
  --dk:#171A18; --dk2:#222624; --hair:rgba(23,24,21,.16); --hair2:rgba(23,24,21,.08); --card:#FBFAF5;
  --disp:"Archivo Black","Arial Black",Impact,sans-serif; --ser:"Instrument Serif",Georgia,serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; --sans:"DM Sans",system-ui,-apple-system,sans-serif;
  --accent: var(--ink); --muted: var(--gr); --line: var(--hair); --soft: var(--min); }
body, gradio-app, .gradio-container { background: var(--paper) !important; }
/* The page is always the light paper design: in a dark-mode browser Gradio would otherwise switch text to near-white. */
.gradio-container, .gradio-container .prose, .gradio-container .prose *, .gradio-container .html-container *,
.req, .req *, .step, .step *, .tbl, .tbl *, .note, .note *, .verdict, .verdict span, .lede, .lede *,
.gradio-container input, .gradio-container label, .gradio-container .wrap, .gradio-container li { color: var(--ink) !important; }
.dark .gradio-container .block, .dark .gradio-container input, .dark .gradio-container .wrap,
.dark .gradio-container .secondary-wrap { background: var(--card) !important; color: var(--ink) !important; }
.hd, .tt, .lbl, .empty, .fine, th, .args { color: var(--gr) !important; }
.banner.live { color: var(--chr) !important; } .banner.recorded { color: var(--ver-t) !important; }
.verdict b { color: var(--ver) !important; } mark.ack { color: var(--ink) !important; }
.hero .red { color: var(--ver) !important; } .lede a { color: var(--ver-t) !important; }
.gradio-container { max-width: 1240px !important; color: var(--ink); font-family: var(--sans) !important; }
footer { display: none !important; }
.topbar { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; border-bottom: 1px solid var(--hair); padding: 6px 0 12px; margin-bottom: 28px; color: var(--ink); }
.hero { font-family: var(--disp); text-transform: uppercase; font-size: clamp(40px, 6.2vw, 78px); line-height: .92;
  letter-spacing: -.01em; margin: 0 0 18px; color: var(--ink); }
.hero .ok { font-family: var(--ser); font-style: italic; text-transform: none; font-weight: 400; letter-spacing: 0;
  text-decoration: line-through; text-decoration-color: var(--ver); text-decoration-thickness: .08em; }
.hero .red { color: var(--ver); }
.lede { color: var(--ink); font-size: 16.5px; line-height: 1.55; max-width: 860px; margin: 0 0 22px; }
.lede .pull { font-family: var(--ser); font-style: italic; font-size: 24px; line-height: 1.2; display: block; margin-bottom: 10px; }
.lede a { color: var(--ver-t); }
.guide { background: var(--dk); color: var(--paper); border-radius: 2px; padding: 18px 22px; margin: 6px 0 18px; }
.guide .eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--chr); margin-bottom: 8px; }
.gradio-container .prose .guide, .gradio-container .prose .guide li, .gradio-container .prose .guide p,
.gradio-container .prose .guide i, .gradio-container .prose .guide div, .gradio-container .prose .guide span,
.gradio-container .guide, .gradio-container .guide * { color: var(--paper) !important; }
.gradio-container .prose .guide b, .gradio-container .guide b { color: #fff !important; }
.gradio-container .prose .guide code, .gradio-container .guide code { color: var(--chr) !important; background: transparent; }
.gradio-container .prose .guide .eyebrow, .gradio-container .guide .eyebrow { color: var(--chr) !important; }
.gradio-container .prose .guide .legend .l-live, .gradio-container .guide .legend .l-live { color: var(--ink) !important; }
.gradio-container .prose .guide .legend .l-rec, .gradio-container .guide .legend .l-rec { color: var(--ver) !important; }
.guide ol { margin: 6px 0 10px 18px; padding: 0; } .guide li { margin: 3px 0; }
.guide li::marker { font-family: var(--mono); color: var(--chr); }
.legend { margin: 6px 0; font-size: 13.5px; }
.legend span { display: inline-block; padding: 2px 8px; margin-right: 8px; font-family: var(--mono); font-size: 11.5px; letter-spacing: .04em; }
.legend .l-live { background: var(--chr); color: var(--ink); }
.legend .l-scr { border: 1px solid var(--paper); color: var(--paper); }
.legend .l-rec { border: 1px dashed var(--ver); color: var(--ver); }
.banner { padding: 9px 12px; border-radius: 2px; font-family: var(--mono); font-size: 12px; letter-spacing: .02em; }
.banner.live { background: var(--dk); color: var(--chr); }
.banner.scripted { background: var(--card); color: var(--ink); border: 1px solid var(--ink); }
.banner.recorded { background: var(--min); color: var(--ver-t); border: 1px dashed var(--ver); }
h1, h2, h3, h4 { font-family: var(--disp); font-weight: 400; color: var(--ink); text-transform: uppercase; letter-spacing: .01em; }
.req, .step, .tbl, .note { border: 1px solid var(--hair); border-radius: 2px; padding: 10px 12px; margin: 8px 0; background: var(--card); }
.req { background: var(--min); }
.hd, .tt { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; color: var(--gr); margin-bottom: 4px; }
.step.report { border-left: 3px solid var(--ink); }
.step.raw pre { margin: 0; white-space: pre-wrap; font-size: 12px; font-family: var(--mono); }
.lbl { display: inline-block; width: 42px; color: var(--gr); font-family: var(--mono); font-size: 11px; }
code { font-family: var(--mono); font-size: 12.5px; background: transparent; word-break: break-word; }
mark.ack { background: var(--chr); color: var(--ink); font-weight: 500; padding: 0 3px; border-radius: 1px; }
.note { background: var(--min); }
.verdict { display: flex; gap: 24px; padding: 10px 12px; border: 1px solid var(--ink); border-radius: 2px; margin: 8px 0; background: var(--card); }
.verdict b { font-family: var(--disp); font-weight: 400; font-size: 20px; color: var(--ver); }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 3px 8px 3px 0; border-bottom: 1px solid var(--hair2); vertical-align: top; }
th { font-family: var(--mono); font-weight: 500; font-size: 11px; color: var(--gr); }
table.kv th { width: 150px; }
.empty, .fine { color: var(--gr); font-size: 12px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 12px; }
.cmp { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 16px; }
.cmp h4 { margin: 8px 0 0 0; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.step.gate { border-left: 3px solid var(--ink); background: #EEF3D8; }
.step.blocked { border-left: 3px solid var(--ver); }
.gnote { color: var(--ink); font-size: 12.5px; font-weight: 700; margin: 2px 0; }
.gradio-container label span, .gradio-container .label-wrap span, .gradio-container [data-testid="block-info"] {
  font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: .06em; text-transform: uppercase;
  color: #4A4B45 !important; }
button.primary { font-family: var(--mono) !important; text-transform: uppercase; letter-spacing: .12em; border-radius: 2px !important; }
@media (max-width: 760px) { .cols, .cmp, .grid2 { grid-template-columns: 1fr; grid-auto-flow: row; } .topbar span:nth-child(2) { display: none; } }
"""

THEME = gr.themes.Base(primary_hue="orange", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"]).set(
    button_primary_background_fill="#171815", button_primary_background_fill_hover="#F05236",
    button_primary_text_color="#F1EFE7", button_primary_text_color_hover="#171815",
    body_background_fill="#F1EFE7", background_fill_primary="#FBFAF5", background_fill_secondary="#E3E0D6",
    block_background_fill="#FBFAF5", block_border_color="rgba(23,24,21,.16)", block_radius="2px",
    input_background_fill="#FBFAF5", border_color_primary="rgba(23,24,21,.16)", color_accent="#F05236")

INTRO = f"""
<div class="topbar"><span>ActionProof · live</span><span>a research study</span><span>training AI agents that use tools</span><span>2026</span></div>
<h1 class="hero">The tool said <span class="ok">ok.</span><br>Run it. <span class="red">Check the database.</span></h1>
<p class="lede"><span class="pull">The same request and the same "ok" can hide different realities.</span>
Pick a task and a reality, run an agent,
and only after the run see what the databases actually hold. All tools are simulated; nothing touches a real system.
The registered study, with every recorded episode and result: <a href="{REPLAY_URL}" target="_blank">replay site</a>. This page is a
curated live illustration of one task and one reality at a time, on ProcureSim tasks from the P1 test; it is not the benchmark.
The hand-written reconciliation script (<i>SCRIPTED: transaction-aware oracle</i>) also gets these tasks right, and on the
benchmark it still beats every learned agent.</p>
<div class="guide"><div class="eyebrow">Judge mode · preselected</div><b>About 70 seconds once the GPU is warm; the first run after a sleep takes longer.</b>
<ol><li>Press <b>Run</b>. Left: a script that trusts the reply. Right: the trained Cohere agent, a real model call per step.</li>
<li>Watch the timeline: the same request and the same <code>"status": "ok"</code> reply for both.</li>
<li>Read the reveal below: the database rows, the verifier's verdict per step, and each agent's decision record. Download the run as JSON.</li></ol>
<div class="legend"><span class="l-live">LIVE MODEL RUN</span><span class="l-scr">SCRIPTED</span><span class="l-rec">RECORDED MODEL RUN</span>
If the GPU is unavailable, the banner says so and shows the recorded run of the same model, task and reality.</div>
<b>New: the Recovery Gate.</b> A small evidence-only controller between an agent and the tools: it verifies before a
report is accepted, reads before a repeated write, blocks wrong repairs and hands off when a record cannot be read.
Pick <i>Gate demo</i> (no GPU, a few seconds) to see the same reply-trusting script without and with it.</div>
"""


def build_ui(stream=model_stream) -> gr.Blocks:
    tasks = task_choices()
    with gr.Blocks(theme=THEME, css=CSS, title="ActionProof live") as demo:
        gr.HTML(INTRO)
        state = gr.State({"runs": 0})
        with gr.Row():
            task = gr.Dropdown(list(tasks), value=list(tasks)[0], label="Task (curated test tasks with a pay step)")
            reality = gr.Dropdown(list(REALITIES), value="Reality C",
                                  label="Reality (what the payment system really does; revealed after the run)")
            agent = gr.Dropdown([(v, k) for k, v in AGENTS.items()], value="judge", label="Agent")
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
        export = gr.File(label="This run as JSON (run id, model, revision, decoding, calls, replies, verdicts)")

        def go(agent_key, task_label, reality_key, st):
            try:
                for ui in run_session(agent_key, tasks[task_label], reality_key, st, stream):
                    path = None
                    if ui.get("export"):
                        fd, path = tempfile.mkstemp(prefix=f"actionproof-{ui['export']['run_id']}-", suffix=".json")
                        with os.fdopen(fd, "w") as f:
                            json.dump(ui["export"], f, indent=1, default=str)
                    yield (ui["banner_l"], ui["left"], ui["banner_r"], ui["right"], ui["reveal"], ui["status"], st, path)
            except Exception as e:  # a failure with no recording to fall back on
                yield ("", "", "", "", "", f"Run failed: {type(e).__name__}: {e}. Scripted agents need no GPU; "
                       f"the replay site has every recorded episode.", st, None)

        run.click(go, [agent, task, reality, state], [ban_l, left, ban_r, right, reveal, status, state, export],
                  concurrency_limit=1, concurrency_id="run")
    return demo


demo = build_ui()
demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
