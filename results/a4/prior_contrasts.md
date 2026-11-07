
### val
| arm | run | n | VSC [Wilson95] | FCR (k/G0) | post-ack refund check | post-ack any write | verify calls/ep | DSR | IVR | report rate |
|---|---|---|---|---|---|---|---|---|---|---|
| init (RFT v2) | rft_tiny_v2 | 100 | 0.59 [0.492,0.681] | 35/36=0.97 | 0/77=0.00 | 0.00 | 2.43 | 0.0 | 0.06 | 0.99 |
| tool_trust | grpo_tiny_v2_tool_trust_s0 | 100 | 0.65 [0.553,0.736] | 35/35=1.00 | 0/91=0.00 | 0.00 | 2.06 | 0.0 | 0.06 | 1.0 |
| tool_trust | grpo_tiny_v2_tool_trust_s1 | 100 | 0.64 [0.542,0.727] | 35/36=0.97 | 0/87=0.00 | 0.00 | 2.19 | 0.0 | 0.06 | 0.99 |
| llm_judge | grpo_tiny_v2_llm_judge_s0 | 100 | 0.64 [0.542,0.727] | 35/36=0.97 | 0/77=0.00 | 0.00 | 2.0 | 0.0 | 0.06 | 1.0 |
| outcome | grpo_tiny_v2_outcome_s0 | 100 | 0.86 [0.779,0.915] | 13/13=1.00 | 76/77=0.99 | 0.89 | 2.17 | 0.0 | 0.06 | 1.0 |
| outcome | grpo_tiny_v2_outcome_s1 | 100 | 0.86 [0.779,0.915] | 13/13=1.00 | 76/77=0.99 | 0.89 | 2.17 | 0.0 | 0.06 | 1.0 |
| pcstr | grpo_tiny_v2_pcstr_s0 | 100 | 0.87 [0.790,0.922] | 13/13=1.00 | 76/77=0.99 | 0.89 | 2.17 | 0.0 | 0.06 | 1.0 |
| pcstr_d | grpo_tiny_v2_pcstr_d_s0 | 100 | 0.87 [0.790,0.922] | 13/13=1.00 | 76/77=0.99 | 0.89 | 2.17 | 0.0 | 0.06 | 1.0 |
| pcstr_d | grpo_tiny_v2_pcstr_d_s1 | 100 | 0.87 [0.790,0.922] | 13/13=1.00 | 76/77=0.99 | 0.89 | 2.17 | 0.0 | 0.06 | 1.0 |
| pcstr_c | grpo_tiny_v2_pcstr_c_s0 | 100 | 0.87 [0.790,0.922] | 13/13=1.00 | 44/77=0.57 | 0.48 | 1.7 | 0.0 | 0.06 | 1.0 |

### test_iid
| arm | run | n | VSC [Wilson95] | FCR (k/G0) | post-ack refund check | post-ack any write | verify calls/ep | DSR | IVR | report rate |
|---|---|---|---|---|---|---|---|---|---|---|
| init (RFT v2) | rft_tiny_v2 | 200 | 0.555 [0.486,0.622] | 77/80=0.96 | 0/154=0.00 | 0.01 | 2.4 | 0.0 | 0.03 | 0.985 |
| tool_trust | grpo_tiny_v2_tool_trust_s0 | 200 | 0.61 [0.541,0.675] | 77/78=0.99 | 0/175=0.00 | 0.00 | 2.0 | 0.0 | 0.03 | 0.995 |
| tool_trust | grpo_tiny_v2_tool_trust_s1 | 200 | 0.605 [0.536,0.670] | 78/79=0.99 | 0/173=0.00 | 0.00 | 2.1 | 0.0 | 0.03 | 0.995 |
| llm_judge | grpo_tiny_v2_llm_judge_s0 | 200 | 0.61 [0.541,0.675] | 77/78=0.99 | 0/155=0.00 | 0.00 | 1.95 | 0.0 | 0.03 | 1.0 |
| outcome | grpo_tiny_v2_outcome_s0 | 200 | 0.815 [0.755,0.863] | 30/30=1.00 | 154/155=0.99 | 0.93 | 2.14 | 0.0 | 0.03 | 1.0 |
| outcome | grpo_tiny_v2_outcome_s1 | 200 | 0.835 [0.777,0.880] | 30/30=1.00 | 153/155=0.99 | 0.92 | 2.13 | 0.0 | 0.03 | 1.0 |
| pcstr | grpo_tiny_v2_pcstr_s0 | 200 | 0.85 [0.794,0.893] | 30/30=1.00 | 152/155=0.98 | 0.91 | 2.13 | 0.0 | 0.03 | 1.0 |
| pcstr_d | grpo_tiny_v2_pcstr_d_s0 | 200 | 0.85 [0.794,0.893] | 30/30=1.00 | 152/155=0.98 | 0.91 | 2.13 | 0.0 | 0.03 | 1.0 |
| pcstr_d | grpo_tiny_v2_pcstr_d_s1 | 200 | 0.85 [0.794,0.893] | 30/30=1.00 | 153/155=0.99 | 0.92 | 2.13 | 0.0 | 0.03 | 1.0 |
| pcstr_c | grpo_tiny_v2_pcstr_c_s0 | 200 | 0.85 [0.794,0.893] | 30/30=1.00 | 93/155=0.60 | 0.53 | 1.72 | 0.0 | 0.03 | 1.0 |

### test_ood_fault
| arm | run | n | VSC [Wilson95] | FCR (k/G0) | post-ack refund check | post-ack any write | verify calls/ep | DSR | IVR | report rate |
|---|---|---|---|---|---|---|---|---|---|---|
| init (RFT v2) | rft_tiny_v2 | 200 | 0.685 [0.618,0.745] | 63/63=1.00 | 0/172=0.00 | 0.00 | 2.65 | 0.0 | 0.06 | 1.0 |
| tool_trust | grpo_tiny_v2_tool_trust_s0 | 200 | 0.665 [0.597,0.727] | 67/67=1.00 | 0/172=0.00 | 0.00 | 2.21 | 0.0 | 0.07 | 1.0 |
| tool_trust | grpo_tiny_v2_tool_trust_s1 | 200 | 0.675 [0.607,0.736] | 65/65=1.00 | 0/172=0.00 | 0.00 | 2.36 | 0.0 | 0.07 | 1.0 |
| llm_judge | grpo_tiny_v2_llm_judge_s0 | 200 | 0.68 [0.612,0.741] | 64/64=1.00 | 0/172=0.00 | 0.00 | 2.01 | 0.0 | 0.065 | 1.0 |
| outcome | grpo_tiny_v2_outcome_s0 | 200 | 0.815 [0.755,0.863] | 37/37=1.00 | 187/187=1.00 | 0.97 | 2.33 | 0.0 | 0.06 | 1.0 |
| outcome | grpo_tiny_v2_outcome_s1 | 200 | 0.81 [0.750,0.858] | 38/38=1.00 | 184/184=1.00 | 0.96 | 2.29 | 0.0 | 0.055 | 1.0 |
| pcstr | grpo_tiny_v2_pcstr_s0 | 200 | 0.81 [0.750,0.858] | 38/38=1.00 | 182/182=1.00 | 0.95 | 2.32 | 0.0 | 0.055 | 1.0 |
| pcstr_d | grpo_tiny_v2_pcstr_d_s0 | 200 | 0.815 [0.755,0.863] | 37/37=1.00 | 186/186=1.00 | 0.97 | 2.33 | 0.0 | 0.055 | 1.0 |
| pcstr_d | grpo_tiny_v2_pcstr_d_s1 | 200 | 0.81 [0.750,0.858] | 38/38=1.00 | 183/183=1.00 | 0.95 | 2.33 | 0.0 | 0.055 | 1.0 |
| pcstr_c | grpo_tiny_v2_pcstr_c_s0 | 200 | 0.805 [0.745,0.854] | 39/39=1.00 | 125/182=0.69 | 0.64 | 1.97 | 0.0 | 0.06 | 1.0 |

### contrasts (VSC), test_iid ∪ test_ood_fault = 400 tasks
| contrast | seed pairing | Δ | 95% CI | discordant (A>B / B>A) | exact McNemar p |
|---|---|---|---|---|---|
| pcstr_d − tool_trust | s0 vs s0 | +0.1950 | [+0.1550,+0.2350] | 79/1 | 1.3e-22 |
| pcstr_d − tool_trust | s1 vs s1 | +0.1900 | [+0.1525,+0.2300] | 77/1 | 5.2e-22 |
| pcstr_d − tool_trust | hierarchical (all seeds) | +0.1925 | [+0.1537,+0.2325] | | |
| outcome − tool_trust | s0 vs s0 | +0.1775 | [+0.1350,+0.2200] | 79/8 | 8.4e-16 |
| outcome − tool_trust | s1 vs s1 | +0.1825 | [+0.1425,+0.2225] | 77/4 | 1.4e-18 |
| outcome − tool_trust | hierarchical (all seeds) | +0.1800 | [+0.1388,+0.2225] | | |
| pcstr_d − outcome | s0 vs s0 | +0.0175 | [+0.0050,+0.0325] | 7/0 | 0.016 |
| pcstr_d − outcome | s1 vs s1 | +0.0075 | [+0.0000,+0.0175] | 3/0 | 0.25 |
| pcstr_d − outcome | hierarchical (all seeds) | +0.0125 | [+0.0037,+0.0238] | | |
| llm_judge − tool_trust | s0 vs s0 | +0.0075 | [-0.0025,+0.0175] | 4/1 | 0.38 |
| llm_judge − tool_trust | hierarchical (all seeds) | +0.0063 | [-0.0025,+0.0150] | | |
| pcstr − outcome | s0 vs s0 | +0.0150 | [+0.0025,+0.0300] | 7/1 | 0.07 |
| pcstr − outcome | hierarchical (all seeds) | +0.0112 | [+0.0025,+0.0225] | | |
| pcstr_c − pcstr_d | s0 vs s0 | -0.0050 | [-0.0125,+0.0000] | 0/2 | 0.5 |
| pcstr_c − pcstr_d | hierarchical (all seeds) | -0.0037 | [-0.0100,+0.0000] | | |
| tool_trust − init (RFT v2) | s0 vs s0 | +0.0175 | [+0.0000,+0.0375] | 11/4 | 0.12 |
| tool_trust − init (RFT v2) | s1 vs s0 | +0.0200 | [+0.0050,+0.0375] | 10/2 | 0.039 |
| tool_trust − init (RFT v2) | hierarchical (all seeds) | +0.0187 | [+0.0013,+0.0375] | | |
| outcome − init (RFT v2) | s0 vs s0 | +0.1950 | [+0.1525,+0.2400] | 86/8 | 1.2e-17 |
| outcome − init (RFT v2) | s1 vs s0 | +0.2025 | [+0.1625,+0.2450] | 85/4 | 8.3e-21 |
| outcome − init (RFT v2) | hierarchical (all seeds) | +0.1988 | [+0.1550,+0.2425] | | |
