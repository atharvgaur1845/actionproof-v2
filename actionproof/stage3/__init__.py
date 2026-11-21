"""Stage 3: algorithms for evidence-grounded execution (docs/stage3/RESEARCH_REPORT.md).

Isolated from every frozen Stage 2 module: ProcureSim (`integrity/sim`), the A12 trainer and rollout function are
imported, never edited. New pieces: a ratio-aware environment (`env`), reward variants (`rewards`), recovery-start
task pools (`tasks`), a rollout function that replays a scripted prefix before the policy acts (`rollout`), and one
training entry point (`train`).
"""
