# Research Notebook

---

## Basic information

| Item | Details |
|---|---|
| Topic | Generalization of image-based citrus disease classification models |
| Researcher | |
| Supervising teacher | |
| Start date | |
| Paper reproduced | Devora-Guadarrama et al. (2025), *Computers*, 14(11), 500 |

---

## Experiment environment

These details are printed at the top of the `run_1_prepare.py` output and are
copied here unchanged.

| Item | Details |
|---|---|
| Computer | |
| Operating system | |
| Python version | |
| PyTorch version | |
| GPU | |
| Random seed | 42 |

---

## Data provenance

| | Source | URL / location | Date | Images | Labels confirmed by |
|---|---|---|---|---|---|
| A-1 | Citrus Leaves Dataset (Rauf et al., 2019, Mendeley) | <https://github.com/ai-agriculture-circuits-and-systems/citrus_leaves> | Published 2019 | 420 (healthy 58 / canker 163 / greening 199) | |
| A-2 | Orange Leaves Images Dataset for HLB (Mendeley, 2025) | <https://data.mendeley.com/datasets/jgkh2jxbwt/1> | Published 2025 | healthy 129 / greening 195 | |
| B-1 | Plant Village Orange — Huanglongbing | <https://github.com/ai-agriculture-circuits-and-systems/Plant_Village_Orange> | PlantVillage release | greening 5,507 | |
| B-2 | kaku321 / citrus-plant-disease (Kaggle) | <https://www.kaggle.com/datasets/kaku321/citrus-plant-disease> | Unknown | healthy 100 / canker 52 / greening 69 | |
| B-3 | CitrusUAT (Gómez-Flores et al., 2024) | <https://zenodo.org/records/8294078> | Published 2024 | healthy 100 / greening 43 | HLB confirmed by qPCR (original paper) |

The cleaning steps (removal of duplicates and augmented copies) are described
below the provenance table in `01_data_collection_guide.md`.

Label verification record (expert confirmation of the diagnoses):

| Date | Institution / person | Method | Images | Result |
|---|---|---|---|---|
| | | | | |

---

## Experiment log

---

### Experiment #1

**Date**:
**Purpose**:

**Settings** (only the values changed in `config.yaml`)

| Item | Value |
|---|---|
| model.name | |
| train.epochs | |
| train.batch_size | |
| train.learning_rate | |
| seed | 42 |

**Results**

| Condition | Accuracy | F1 | MCC | Images |
|---|---|---|---|---|
| Reproduction (same source) | | | | |
| Generalization (new images) | | | | |
| Reproduction + background removed | | | | |
| Generalization + background removed | | | | |

**Generalization gap**: accuracy ___ percentage points, F1 ___

**Leaf focus**

| Source | Mean focus | Valid samples |
|---|---|---|
| source_A | | / |
| source_B | | / |

**Observations**

-
-

**Interpretation** (why this result happened)

-

**Next**

-

---

### Experiment #2 — does more training data close the gap?

**Date**: 2026-09-26
**Purpose**: Test whether the failure to generalize comes from set A being too
small, or from all of its images coming from one source.

**Settings**: densenet121, seed 42, everything else as in `config.yaml`. Two
arms, both evaluated on the same unchanged source_B (5,871 images).

- Arm 1, quantity: trained on 25 / 50 / 75 / 100% of set A
  (`run_2_train.py --train-fraction`), one source only.
- Arm 2, diversity: set A expanded with a second source
  (Orange Leaves HLB 2025), 294 → 521 training images.

**Results**

| Arm | Train images | A test acc | B balanced acc | B MCC |
|---|---|---|---|---|
| 25% of A | 73 | 96.8% | 0.497 | 0.016 |
| 50% of A | 147 | 95.2% | 0.432 | 0.009 |
| 75% of A | 221 | 100.0% | 0.415 | 0.037 |
| 100% of A | 294 | 100.0% | 0.556 | 0.067 |
| A + second source | 521 | 98.2% | 0.623 | 0.095 |

source_B accuracy by origin, one source (100% of A) vs two sources:

| Class / origin | One source | Two sources |
|---|---|---|
| healthy / CitrusUAT | 81.0% | 100.0% |
| healthy / kaku321 | 67.0% | 98.0% |
| greening / kaku321 | 7.2% | 46.4% |
| greening / CitrusUAT | 0.0% | 23.3% |
| greening / PlantVillage | 0.4% | 6.6% |
| canker / kaku321 (no new data) | 92.3% | 80.8% |

**Observations**

- Four times as many images from one source did not improve cross-source
  balanced accuracy. It moved between 0.415 and 0.556 with no trend, and
  neighbouring points differ as much as the extremes do.
- Same-source accuracy reached 100% at 75% of the data, so the models were not
  short of data for the task they were trained on.
- Adding 227 images from a second source raised balanced accuracy above every
  quantity arm, and every class that received new-source images improved.
- Canker, the one class with no new images, fell.

**Interpretation**

- The bottleneck is the number of sources, not the number of images.
- Greening recall on source_B is still 0.072 overall, because PlantVillage
  supplies 5,507 of the 5,619 greening images and improved least.

**Next**

- Repeat both arms with several seeds. Every point here is a single seed and
  the differences are of the size that seed noise alone could produce.
- Find a second source for canker, so that every class has at least two.

---

### Experiment #3

(same template as above)

---

## Failure log

What didn't work, what caused errors, and what was abandoned.
**This is the most valuable part of the notebook.**

| Date | What was tried | How it failed | How it was resolved (or why it was abandoned) |
|---|---|---|---|
| 2026-09-11 | Kaggle `dtrilsbeek/citrus-leaves-prepared`, downloaded with `kagglehub`, as a source_B candidate | Its class counts (58/163/204) are identical to source_A, so it is a repackaged copy of the same source | Not used as B. Kept only in `data/datasets/citrus-leaves-prepared/` and not added to the pipeline |
| 2026-09-11 | First generalization run (efficientnet_b0, resnet50) with 19,030 source_B images | Half of the Plant Village Orange greening images were byte-identical copies (`_1`), and kaku321 held 48–56 augmented copies of each photo. Counts were inflated, and the models were tested on flipped, mirror-padded images unlike real photos | Results discarded. B was rebuilt with copies removed and one original per block (5,871 images), and the run was repeated (2026-09-18) |
| 2026-09-18 | Checked source_A for internal duplicates with a perceptual hash | Found 5 re-saved copies of the same photos among A's (Citrus Leaves) greening images. Copies split across train and test inflate reproduction accuracy | 5 removed (425 → 420). MD5 didn't catch them, so each was confirmed visually |

---

## Ideas and questions

Questions that come up during the research are noted here as they arise. They
become the discussion section of the paper.

-
-

---

## Checklist (required)

This checklist adapts the reproducibility checklist of Pineau et al. (2021) and
the data leakage checks of Kapoor & Narayanan (2023) to this study. Each item
is checked before the paper is written.

- [ ] Exact model name and version
- [ ] Statement that the random seed was fixed, and its value
- [ ] Data sources, image counts and per-class distribution
- [ ] Train/validation/test ratio and split method (stratified)
- [ ] **Explicit statement that source_B was never used for training**
- [ ] **Explicit statement that A and B were checked for duplicates**
- [ ] Who confirmed the labels, and how
- [ ] Precision, recall, F1 and MCC reported alongside accuracy
- [ ] Per-class results reported, not only the overall average
- [ ] Experiment environment (Python, PyTorch, GPU)
- [ ] Any failed or excluded attempts, and why
