# demo/: sources of the replay site (not a Space README)

This folder holds the *sources* of the replay website: `site_template.html`, `site_upgrade.js`, `data.json`,
`replays/`, `theme.css` and the build helpers. The built page is `demo/static/index.html`, deployed as the static
Space https://huggingface.co/spaces/atharvgaur18/actionproof. It contains recorded episodes only: no model call
happens there.

The **live** Space (a real Cohere model per run) is `space_live/`, deployed as
https://huggingface.co/spaces/atharvgaur18/actionproof-live.

`app.py` here is the Stage 1 Gradio demo (Nov 13), kept for history. It is not deployed.

How the demo is run for judges: `docs/submission/DEMO_RUN_OF_SHOW.md`.
