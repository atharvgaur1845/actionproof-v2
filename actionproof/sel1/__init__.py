"""SEL1 (PREREGISTRATION_SEL1.md): can demonstrations teach selective escalation under hidden commit outcomes?

Built on the frozen TXN v1 env (`actionproof/txn/`), which is not modified:
- `env.py`: a transient read outage (a read tool fails for its first n agent calls, then recovers);
- `tasks.py`: the training pools and the sealed test split;
- `policy.py`: B5, B4 with one change: an unavailable read is re-read up to three times before escalating;
- `build.py`: SFT rows for the two arms (SEL and CTL), rendered exactly as `scripts/p1_eval.py` renders an episode.
"""
