"""Lite parser/prompt must agree with AgentTune's on real model outputs (demo fidelity)."""
import glob, json, sys
from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOLS_FALLBACK_PROMPT
from agenttune.agentic.rollout_engines.tool_call_parse import extract_tool_calls_from_text as at_parse
from actionproof.rollout import lite_format as L

assert L.FALLBACK_PROMPT == DEFAULT_TOOLS_FALLBACK_PROMPT, "prompt text drifted from AgentTune"
n = agree = 0
for f in sys.argv[1:] or glob.glob("results/eval/**/*.jsonl", recursive=True)[:6]:
    for line in open(f):
        r = json.loads(line)
        for m in r.get("conversation", []):
            if m["role"] != "assistant":
                continue
            a, b = at_parse(m["content"]) or [], L.extract_tool_calls_from_text(m["content"]) or []
            norm = lambda cs: [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")) for c in cs]
            n += 1
            agree += norm(a) == norm(b)
print(json.dumps({"assistant_turns": n, "agree": agree, "rate": agree / max(n, 1)}))
