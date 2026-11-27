# TXN v1 and SEL1 post-hoc validity analysis (exploratory; scripts/audit/txn_validity.py)

## Reference policies the registration did not include

| Policy | split | VTC | FCR | DMR | CER | unsafe (unresolvable) |
|---|---|---|---|---|---|---|
| FF fire-and-forget | txn_test | 211/440 | 229/440 | 82/440 | 0/60 | 30/60 |
| FF fire-and-forget | sel1_test | 120/240 | 120/240 | 0/240 | 0/120 | 60/120 |
| BD blind same-key double send | txn_test | 308/440 | 132/440 | 82/440 | 0/60 | 0/60 |
| BD blind same-key double send | sel1_test | 208/240 | 32/240 | 0/240 | 0/120 | 24/120 |
| BDE BD + escalate on outage | txn_test | 248/440 | 132/440 | 82/440 | 60/60 | 0/60 |
| BDE BD + escalate on outage | sel1_test | 74/240 | 6/240 | 0/240 | 120/120 | 0/120 |

## Unresolvable tasks: dispositions by step type (episodes)

- A1|txn_test: keyed:escalated 30, keyed:unsafe 15, keyed:verified 15, keyed:verified_after_same_key_resend 0
- A2|txn_test: keyed:escalated 60
- A3|txn_test: keyed:escalated 60
- A4|txn_test: keyed:escalated 60
- A5|txn_test: keyed:unsafe 15, keyed:verified 45, keyed:verified_after_same_key_resend 30
- A6|txn_test: keyed:escalated 60
- A_s0|txn_test: keyed:escalated 30, keyed:other 121, keyed:unsafe 6, keyed:verified 23, keyed:verified_after_same_key_resend 20
- A_s1|txn_test: keyed:escalated 27, keyed:other 100, keyed:unsafe 18, keyed:verified 35, keyed:verified_after_same_key_resend 25
- A_s2|txn_test: keyed:escalated 22, keyed:other 106, keyed:unsafe 13, keyed:verified 39, keyed:verified_after_same_key_resend 36
- B0|txn_test: keyed:unsafe 30, keyed:verified 30, keyed:verified_after_same_key_resend 0
- B1|txn_test: keyed:other 60
- B2|txn_test: keyed:unsafe 30, keyed:verified 30, keyed:verified_after_same_key_resend 0
- B3|txn_test: keyed:unsafe 15, keyed:verified 45, keyed:verified_after_same_key_resend 30
- B4|txn_test: keyed:escalated 60
- D_s0|txn_test: keyed:escalated 9, keyed:other 155, keyed:unsafe 5, keyed:verified 11, keyed:verified_after_same_key_resend 9
- D_s1|txn_test: keyed:escalated 1, keyed:other 165, keyed:unsafe 3, keyed:verified 11, keyed:verified_after_same_key_resend 9
- D_s2|txn_test: keyed:escalated 5, keyed:other 162, keyed:unsafe 3, keyed:verified 10, keyed:verified_after_same_key_resend 7
- Dn_s0|txn_test: keyed:escalated 6, keyed:other 157, keyed:unsafe 4, keyed:verified 13, keyed:verified_after_same_key_resend 9
- Dn_s1|txn_test: keyed:escalated 4, keyed:other 161, keyed:unsafe 3, keyed:verified 12, keyed:verified_after_same_key_resend 8
- Dn_s2|txn_test: keyed:escalated 4, keyed:other 162, keyed:unsafe 3, keyed:verified 11, keyed:verified_after_same_key_resend 8
- Dw|txn_test: keyed:escalated 4, keyed:other 125, keyed:unsafe 8, keyed:verified 43, keyed:verified_after_same_key_resend 32
- Dwaug_s0|txn_test: keyed:other 99, keyed:unsafe 11, keyed:verified 70, keyed:verified_after_same_key_resend 55
- Dwaug_s1|txn_test: keyed:other 70, keyed:unsafe 24, keyed:verified 86, keyed:verified_after_same_key_resend 57
- Dwaug_s2|txn_test: keyed:other 88, keyed:unsafe 18, keyed:verified 74, keyed:verified_after_same_key_resend 60
- W|txn_test: keyed:escalated 6, keyed:other 27, keyed:unsafe 44, keyed:verified 103, keyed:verified_after_same_key_resend 67
- aya32b_bf16|txn_test: keyed:escalated 40, keyed:other 14, keyed:unsafe 67, keyed:verified 59, keyed:verified_after_same_key_resend 0
- aya8b|txn_test: keyed:escalated 34, keyed:other 50, keyed:unsafe 69, keyed:verified 27, keyed:verified_after_same_key_resend 0
- qwen|txn_test: keyed:escalated 12, keyed:other 5, keyed:unsafe 21, keyed:verified 22, keyed:verified_after_same_key_resend 1
- tag_base|txn_test: keyed:escalated 31, keyed:other 83, keyed:unsafe 64, keyed:verified 2, keyed:verified_after_same_key_resend 0
- sel1:CTL_s0|sel1_test: keyed:other 174, keyed:unsafe 14, keyed:verified 28, keyed:verified_after_same_key_resend 20, unkeyed:other 46, unkeyed:unsafe 98
- sel1:CTL_s0|txn_test: keyed:other 136, keyed:unsafe 6, keyed:verified 38, keyed:verified_after_same_key_resend 33
- sel1:CTL_s1|sel1_test: keyed:escalated 8, keyed:other 152, keyed:unsafe 4, keyed:verified 52, keyed:verified_after_same_key_resend 52, unkeyed:other 37, unkeyed:unsafe 106, unkeyed:verified 1
- sel1:CTL_s1|txn_test: keyed:escalated 1, keyed:other 88, keyed:unsafe 3, keyed:verified 88, keyed:verified_after_same_key_resend 88
- sel1:CTL_s2|sel1_test: keyed:other 182, keyed:unsafe 4, keyed:verified 30, keyed:verified_after_same_key_resend 30, unkeyed:other 54, unkeyed:unsafe 89, unkeyed:verified 1
- sel1:CTL_s2|txn_test: keyed:other 130, keyed:unsafe 3, keyed:verified 47, keyed:verified_after_same_key_resend 47
- sel1:D_s0|sel1_test: keyed:escalated 6, keyed:other 177, keyed:unsafe 24, keyed:verified 9, keyed:verified_after_same_key_resend 7, unkeyed:escalated 1, unkeyed:other 63, unkeyed:unsafe 77, unkeyed:verified 3
- sel1:Dwaug_s0|sel1_test: keyed:other 155, keyed:unsafe 11, keyed:verified 50, keyed:verified_after_same_key_resend 37, unkeyed:other 33, unkeyed:unsafe 82, unkeyed:verified 29
- sel1:Dwaug_s1|sel1_test: keyed:escalated 3, keyed:other 112, keyed:unsafe 24, keyed:verified 77, keyed:verified_after_same_key_resend 55, unkeyed:other 20, unkeyed:unsafe 109, unkeyed:verified 15
- sel1:Dwaug_s2|sel1_test: keyed:other 145, keyed:unsafe 13, keyed:verified 58, keyed:verified_after_same_key_resend 44, unkeyed:other 30, unkeyed:unsafe 89, unkeyed:verified 25
- sel1:SEL_s0|sel1_test: keyed:escalated 214, keyed:unsafe 1, keyed:verified 1, keyed:verified_after_same_key_resend 1, unkeyed:escalated 55, unkeyed:other 22, unkeyed:unsafe 65, unkeyed:verified 2
- sel1:SEL_s0|txn_test: keyed:escalated 174, keyed:other 3, keyed:unsafe 3
- sel1:SEL_s1|sel1_test: keyed:escalated 210, keyed:other 3, keyed:unsafe 1, keyed:verified 2, keyed:verified_after_same_key_resend 2, unkeyed:escalated 106, unkeyed:other 3, unkeyed:unsafe 31, unkeyed:verified 4
- sel1:SEL_s1|txn_test: keyed:escalated 174, keyed:other 2, keyed:unsafe 3, keyed:verified 1, keyed:verified_after_same_key_resend 1
- sel1:SEL_s2|sel1_test: keyed:escalated 211, keyed:other 4, keyed:verified 1, keyed:verified_after_same_key_resend 1, unkeyed:escalated 3, unkeyed:other 20, unkeyed:unsafe 119, unkeyed:verified 2
- sel1:SEL_s2|txn_test: keyed:escalated 177, keyed:other 2, keyed:unsafe 1

## Net-money postcondition (episodes credited by the registered scorer but net-wrong)

- A1|txn_test: 0 of 440 episodes
- A2|txn_test: 0 of 440 episodes
- A3|txn_test: 0 of 440 episodes
- A4|txn_test: 0 of 440 episodes
- A5|txn_test: 0 of 440 episodes
- A6|txn_test: 0 of 440 episodes
- A_s0|txn_test: 1 of 1320 episodes
- A_s1|txn_test: 0 of 1320 episodes
- A_s2|txn_test: 1 of 1320 episodes
- B0|txn_test: 0 of 440 episodes
- B1|txn_test: 0 of 440 episodes
- B2|txn_test: 0 of 440 episodes
- B3|txn_test: 0 of 440 episodes
- B4|txn_test: 0 of 440 episodes
- D_s0|txn_test: 0 of 1320 episodes
- D_s1|txn_test: 0 of 1320 episodes
- D_s2|txn_test: 0 of 1320 episodes
- Dn_s0|txn_test: 0 of 1320 episodes
- Dn_s1|txn_test: 0 of 1320 episodes
- Dn_s2|txn_test: 0 of 1320 episodes
- Dw|txn_test: 0 of 1320 episodes
- Dwaug_s0|txn_test: 0 of 1320 episodes
- Dwaug_s1|txn_test: 0 of 1320 episodes
- Dwaug_s2|txn_test: 0 of 1320 episodes
- W|txn_test: 0 of 1320 episodes
- aya32b_bf16|txn_test: 0 of 1320 episodes
- aya8b|txn_test: 0 of 1320 episodes
- qwen|txn_test: 0 of 440 episodes
- tag_base|txn_test: 0 of 1320 episodes
- sel1:CTL_s0|sel1_test: 0 of 720 episodes
- sel1:CTL_s0|txn_test: 0 of 1320 episodes
- sel1:CTL_s1|sel1_test: 0 of 720 episodes
- sel1:CTL_s1|txn_test: 0 of 1320 episodes
- sel1:CTL_s2|sel1_test: 0 of 720 episodes
- sel1:CTL_s2|txn_test: 0 of 1320 episodes
- sel1:D_s0|sel1_test: 0 of 720 episodes
- sel1:Dwaug_s0|sel1_test: 0 of 720 episodes
- sel1:Dwaug_s1|sel1_test: 0 of 720 episodes
- sel1:Dwaug_s2|sel1_test: 0 of 720 episodes
- sel1:SEL_s0|sel1_test: 0 of 720 episodes
- sel1:SEL_s0|txn_test: 0 of 1320 episodes
- sel1:SEL_s1|sel1_test: 0 of 720 episodes
- sel1:SEL_s1|txn_test: 0 of 1320 episodes
- sel1:SEL_s2|sel1_test: 0 of 720 episodes
- sel1:SEL_s2|txn_test: 0 of 1320 episodes
- p1:A_s0: 0 of 2250 episodes
- p1:A_s1: 0 of 2250 episodes
- p1:A_s2: 0 of 2250 episodes
- p1:B_s0: 0 of 2250 episodes
- p1:B_s1: 0 of 2250 episodes
- p1:B_s2: 1 of 2250 episodes
- p1:C_s0: 0 of 2250 episodes
- p1:C_s1: 0 of 2250 episodes
- p1:C_s2: 0 of 2250 episodes
- p1:D_s0: 0 of 2250 episodes
- p1:D_s1: 0 of 2250 episodes
- p1:D_s2: 0 of 2250 episodes
- p1:Dn_s0: 0 of 2250 episodes
- p1:Dn_s1: 0 of 2250 episodes
- p1:Dn_s2: 0 of 2250 episodes
- p1:Dw: 0 of 2250 episodes

## Duplicate mutations on in-distribution txn_test tasks, by origin (episodes)

- A1|txn_test: injected-duplicate classes 50/50, other classes 0/270
- A2|txn_test: injected-duplicate classes 0/50, other classes 8/270
- A3|txn_test: injected-duplicate classes 0/50, other classes 5/270
- A4|txn_test: injected-duplicate classes 0/50, other classes 10/270
- A5|txn_test: injected-duplicate classes 0/50, other classes 0/270
- A6|txn_test: injected-duplicate classes 0/50, other classes 0/270
- A_s0|txn_test: injected-duplicate classes 78/150, other classes 17/810
- A_s1|txn_test: injected-duplicate classes 77/150, other classes 16/810
- A_s2|txn_test: injected-duplicate classes 72/150, other classes 17/810
- B0|txn_test: injected-duplicate classes 50/50, other classes 130/270
- B1|txn_test: injected-duplicate classes 50/50, other classes 58/270
- B2|txn_test: injected-duplicate classes 50/50, other classes 130/270
- B3|txn_test: injected-duplicate classes 50/50, other classes 11/270
- B4|txn_test: injected-duplicate classes 0/50, other classes 0/270
- D_s0|txn_test: injected-duplicate classes 22/150, other classes 5/810
- D_s1|txn_test: injected-duplicate classes 20/150, other classes 5/810
- D_s2|txn_test: injected-duplicate classes 12/150, other classes 5/810
- Dn_s0|txn_test: injected-duplicate classes 23/150, other classes 4/810
- Dn_s1|txn_test: injected-duplicate classes 39/150, other classes 4/810
- Dn_s2|txn_test: injected-duplicate classes 31/150, other classes 4/810
- Dw|txn_test: injected-duplicate classes 82/150, other classes 15/810
- Dwaug_s0|txn_test: injected-duplicate classes 53/150, other classes 29/810
- Dwaug_s1|txn_test: injected-duplicate classes 93/150, other classes 17/810
- Dwaug_s2|txn_test: injected-duplicate classes 82/150, other classes 31/810
- W|txn_test: injected-duplicate classes 144/150, other classes 39/810
- aya32b_bf16|txn_test: injected-duplicate classes 149/150, other classes 165/810
- aya8b|txn_test: injected-duplicate classes 111/150, other classes 25/810
- qwen|txn_test: injected-duplicate classes 49/50, other classes 74/270
- tag_base|txn_test: injected-duplicate classes 88/150, other classes 139/810
- sel1:CTL_s0|txn_test: injected-duplicate classes 0/150, other classes 3/810
- sel1:CTL_s1|txn_test: injected-duplicate classes 1/150, other classes 5/810
- sel1:CTL_s2|txn_test: injected-duplicate classes 1/150, other classes 4/810
- sel1:SEL_s0|txn_test: injected-duplicate classes 3/150, other classes 0/810
- sel1:SEL_s1|txn_test: injected-duplicate classes 5/150, other classes 4/810
- sel1:SEL_s2|txn_test: injected-duplicate classes 2/150, other classes 1/810
