Workload per ticket (measured, n=200): 7.5 LLM calls, 25,407 input tokens (19,748 cacheable prefix), 247 output tokens. Self-hosted agent VSC = 0.85.
Self-hosted throughput (measured on NVIDIA H200 NVL): 9,600 tickets/GPU-hour (200 tickets in 75 s, batched vLLM).

| Option | $/1k tickets | $/month at 100,000 tickets | Notes |
|---|---|---|---|
| Self-hosted Tiny Aya, HF Jobs H200 | 0.521 | 52.08 | 10.4 GPU-h/month; price: hf jobs hardware (CLI listing) (2026-11-04) |
| Self-hosted Tiny Aya, HF Jobs A100 80GB | 0.260 | 26.04 | 10.4 GPU-h/month; price: hf jobs hardware (CLI listing) (2026-11-04) |
| Self-hosted Tiny Aya, HF Jobs L4 | 0.083 | 8.33 | 10.4 GPU-h/month; price: hf jobs hardware (CLI listing) (2026-11-04) |
| GPT-5.4 Mini API | 6.837 | 683.67 | with prompt caching; https://wavespeed.ai/blog/posts/gpt-5-4-mini-pricing/ (secondary; verify on openai.com/api/pricing before publishing) (2026-11-05) |
| Gemini 3.1 Flash-Lite API | 2.279 | 227.89 | with prompt caching; https://www.morphllm.com/gemini-api-pricing (secondary; verify on ai.google.dev/pricing; cache storage fee excluded) (2026-11-05) |

GPU rows other than the measured class are priced at the measured throughput (an upper bound for slower GPUs); the measured row is exact.
