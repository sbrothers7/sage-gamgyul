# Analysis — 2026-09-24 22:11

## 1. Debate effect per run (R1 weighted vote -> R3 weighted vote)

| run | condition | n | before | after | Δ | wrong→right | right→wrong | McNemar p |
|---|---|---|---|---|---|---|---|---|
| 20260923_1719_main200_citrusA_ev-on | ON (prompt rule) | 190 | 0.368 | 0.389 | +0.021 | 7 | 3 | 0.344 |
| 20260924_0018_main200_citrusA_ev-off | OFF (no rule) | 190 | 0.368 | 0.384 | +0.016 | 5 | 2 | 0.453 |
| 20260924_1014_main200_citrusA_ev-hard | HARD (code-enforced) | 190 | 0.368 | 0.368 | +0.000 | 2 | 2 | 1.000 |
| 20260924_1638_main180_zenodo_ev-on | ON (prompt rule) | 180 | 0.189 | 0.206 | +0.017 | 7 | 4 | 0.549 |

## 2. Conditions compared on the same images (shared R1)

- **ON (prompt rule)** vs **OFF (no rule)**: only-second-right 4, only-first-right 5, McNemar p = 1.000
- **ON (prompt rule)** vs **HARD (code-enforced)**: only-second-right 3, only-first-right 7, McNemar p = 0.344
- **OFF (no rule)** vs **HARD (code-enforced)**: only-second-right 3, only-first-right 6, McNemar p = 0.508

## 3. Individual agents (accuracy R1 → R2 → R3, flips R1→R3)

| run | agent | R1 | R2 | R3 | changed | wrong→right | right→wrong |
|---|---|---|---|---|---|---|---|
| 20260923_1719_main200_citrusA_ev-on | qwen2.5vl:7b | 0.363 | 0.379 | 0.384 | 60 | 26 | 22 |
| 20260923_1719_main200_citrusA_ev-on | qwen2.5vl:3b | 0.363 | 0.363 | 0.347 | 24 | 4 | 7 |
| 20260923_1719_main200_citrusA_ev-on | llava:7b | 0.284 | 0.374 | 0.389 | 76 | 36 | 16 |
| 20260924_0018_main200_citrusA_ev-off | qwen2.5vl:7b | 0.363 | 0.389 | 0.400 | 55 | 25 | 18 |
| 20260924_0018_main200_citrusA_ev-off | qwen2.5vl:3b | 0.363 | 0.363 | 0.326 | 25 | 3 | 10 |
| 20260924_0018_main200_citrusA_ev-off | llava:7b | 0.284 | 0.379 | 0.432 | 84 | 45 | 17 |
| 20260924_1014_main200_citrusA_ev-hard | qwen2.5vl:7b | 0.363 | 0.358 | 0.358 | 1 | 0 | 1 |
| 20260924_1014_main200_citrusA_ev-hard | qwen2.5vl:3b | 0.363 | 0.363 | 0.374 | 21 | 4 | 2 |
| 20260924_1014_main200_citrusA_ev-hard | llava:7b | 0.284 | 0.289 | 0.305 | 11 | 6 | 2 |
| 20260924_1638_main180_zenodo_ev-on | qwen2.5vl:7b | 0.239 | 0.233 | 0.217 | 42 | 3 | 7 |
| 20260924_1638_main180_zenodo_ev-on | llava:7b | 0.167 | 0.189 | 0.211 | 60 | 10 | 2 |
| 20260924_1638_main180_zenodo_ev-on | qwen2.5vl:3b | 0.128 | 0.128 | 0.161 | 66 | 12 | 6 |

## 4. Rule compliance and convergence

| run | opinion changes (R1→R2, R2→R3) | changes with NO new evidence | reverted by code | unanimous R1 | unanimous R3 | self-named influence |
|---|---|---|---|---|---|---|
| 20260923_1719_main200_citrusA_ev-on | 167 | 143 (86%) | - | 12% | 70% | 51 |
| 20260924_0018_main200_citrusA_ev-off | 168 | 144 (86%) | - | 12% | 62% | 60 |
| 20260924_1014_main200_citrusA_ev-hard | 33 | 0 (0%) | 207 | 12% | 23% | 50 |
| 20260924_1638_main180_zenodo_ev-on | 195 | 129 (66%) | 0 | 32% | 57% | 41 |

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

**20260924_1014_main200_citrusA_ev-hard**

- Black Spot (n=50): 0.20 → 0.22  | final answers: Greening 31, Black Spot 11, Healthy 8
- Canker (n=50): 0.00 → 0.00  | final answers: Black Spot 45, Greening 3, Healthy 2
- Greening (n=50): 0.60 → 0.62  | final answers: Greening 31, Healthy 15, Black Spot 4
- Healthy (n=40): 0.75 → 0.70  | final answers: Healthy 28, Greening 11, Black Spot 1

**20260924_1638_main180_zenodo_ev-on**

- Greasy Spot (n=15): 0.53 → 0.87  | final answers: Greasy Spot 13, Healthy 2
- Greening (n=15): 0.27 → 0.20  | final answers: Healthy 4, Greasy Spot 4, Greening 3, Iron Deficiency 3
- Healthy (n=15): 1.00 → 1.00  | final answers: Healthy 15
- Iron Deficiency (n=15): 0.00 → 0.00  | final answers: Healthy 14, Greasy Spot 1
- Leaf Miner (n=15): 0.40 → 0.40  | final answers: Greasy Spot 6, Leaf Miner 6, Red Scale Sequelae 1, Red Scale 1
- Magnesium Deficiency (n=15): 0.07 → 0.00  | final answers: Greasy Spot 9, Healthy 5, Iron Deficiency 1
- Manganese Deficiency (n=15): 0.00 → 0.00  | final answers: Greasy Spot 9, Iron Deficiency 3, Magnesium Deficiency 1, Healthy 1
- Nitrogen Deficiency (n=15): 0.00 → 0.00  | final answers: Healthy 8, Greasy Spot 7
- Red Scale (n=15): 0.00 → 0.00  | final answers: Greasy Spot 13, Red Scale Sequelae 2
- Red Scale Sequelae (n=15): 0.00 → 0.00  | final answers: Greasy Spot 10, Healthy 2, Leaf Miner 1, Iron Deficiency 1
- Texas Mite (n=15): 0.00 → 0.00  | final answers: Healthy 14, Greasy Spot 1
- Zinc Deficiency (n=15): 0.00 → 0.00  | final answers: Healthy 5, Magnesium Deficiency 4, Greening 3, Iron Deficiency 2
