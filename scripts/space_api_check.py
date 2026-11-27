"""Check a deployed live Space through its API: python scripts/space_api_check.py <repo> <agent,agent,...> [reality].
Prints each run's banners, status, reveal head and export summary (gradio_client; uses the local HF token)."""
import json, re, sys, time
from gradio_client import Client
from huggingface_hub import get_token
repo = sys.argv[1]; agents = sys.argv[2].split(",")
try:
    c = Client(repo, hf_token=get_token(), verbose=False)
except TypeError:  # newer gradio_client names the argument `token`
    c = Client(repo, token=get_token(), verbose=False)
info = c.view_api(return_format="dict", print_info=False)
ep = list(info["named_endpoints"])[0]
params = info["named_endpoints"][ep]["parameters"]
task = params[1].get("example_input") or params[1]["parameter_default"]
for ag in agents:
    t0 = time.time()
    out = c.predict(ag, task, sys.argv[3] if len(sys.argv) > 3 else "Reality C", api_name=ep)
    bl, left, br, right, reveal, status, path = out[:6] + (out[-1],)  # session state is not an API output
    strip = lambda h: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h or "")).strip()
    ex = json.load(open(path)) if path else None
    print(f"== {ag} ({time.time()-t0:.0f} s) task={task}")
    print("  banner L:", strip(bl)[:160]); print("  banner R:", strip(br)[:160])
    print("  status:", strip(status)[:200]); print("  reveal:", strip(reveal)[:220])
    if ex:
        print("  export:", ex.get("run_id"), [(s.get("kind"), s.get("model"), (s.get("score") or {}).get("vsc")) for s in ex.get("sides", [ex])][:2], list(ex)[:12])
