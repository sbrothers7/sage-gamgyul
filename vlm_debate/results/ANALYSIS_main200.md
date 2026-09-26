# Analysis — 2026-09-24 10:05

## 1. Debate effect per run (R1 weighted vote -> R3 weighted vote)

| run | condition | n | before | after | Δ | wrong→right | right→wrong | McNemar p |
|---|---|---|---|---|---|---|---|---|
| 20260923_1719_main200_citrusA_ev-on | ON (prompt rule) | 190 | 0.368 | 0.389 | +0.021 | 7 | 3 | 0.344 |
| 20260924_0018_main200_citrusA_ev-off | OFF (no rule) | 190 | 0.368 | 0.384 | +0.016 | 5 | 2 | 0.453 |

## 2. Conditions compared on the same images (shared R1)

- **ON (prompt rule)** vs **OFF (no rule)**: only-second-right 4, only-first-right 5, McNemar p = 1.000

## 3. Individual agents (accuracy R1 → R2 → R3, flips R1→R3)

| run | agent | R1 | R2 | R3 | changed | wrong→right | right→wrong |
|---|---|---|---|---|---|---|---|
| 20260923_1719_main200_citrusA_ev-on | qwen2.5vl:7b | 0.363 | 0.379 | 0.384 | 60 | 26 | 22 |
| 20260923_1719_main200_citrusA_ev-on | qwen2.5vl:3b | 0.363 | 0.363 | 0.347 | 24 | 4 | 7 |
| 20260923_1719_main200_citrusA_ev-on | llava:7b | 0.284 | 0.374 | 0.389 | 76 | 36 | 16 |
| 20260924_0018_main200_citrusA_ev-off | qwen2.5vl:7b | 0.363 | 0.389 | 0.400 | 55 | 25 | 18 |
| 20260924_0018_main200_citrusA_ev-off | qwen2.5vl:3b | 0.363 | 0.363 | 0.326 | 25 | 3 | 10 |
| 20260924_0018_main200_citrusA_ev-off | llava:7b | 0.284 | 0.379 | 0.432 | 84 | 45 | 17 |

## 4. Rule compliance and convergence

| run | opinion changes (R1→R2, R2→R3) | changes with NO new evidence | reverted by code | unanimous R1 | unanimous R3 | self-named influence |
|---|---|---|---|---|---|---|
| 20260923_1719_main200_citrusA_ev-on | 167 | 143 (86%) | - | 12% | 70% | 51 |
| 20260924_0018_main200_citrusA_ev-off | 168 | 144 (86%) | - | 12% | 62% | 60 |

## 5. Per class accuracy (before → after) and what the group answered

**20260923_1719_main200_citrusA_ev-on**

- Black Spot (n=50): 0.20 → 0.28  | final answers: Greening 25, Black Spot 14, Healthy 10, Canker 1
- Canker (n=50): 0.00 → 0.00  | final answers: Black Spot 43, Greening 5, Healthy 2
- Greening (n=50): 0.60 → 0.64  | final answers: Greening 32, Healthy 17, Black Spot 1
- Healthy (n=40): 0.75 → 0.70  | final answers: Healthy 28, Greening 12

**20260924_0018_main200_citrusA_ev-off**

- Black Spot (n=50): 0.20 → 0.22  | final answers: Greening 27, Healthy 11, Black Spot 11, Canker 1
- Canker (n=50): 0.00 → 0.00  | final answers: Black Spot 44, Greening 4, Healthy 2
- Greening (n=50): 0.60 → 0.66  | final answers: Greening 33, Healthy 16, Black Spot 1
- Healthy (n=40): 0.75 → 0.72  | final answers: Healthy 29, Greening 11
