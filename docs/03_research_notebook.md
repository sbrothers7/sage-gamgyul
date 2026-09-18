# Research Notebook

Every experiment is recorded here at the time it is run, because entries
written afterwards miss details.

At a science fair, the notebook is the main evidence that the student actually
did the research. Failed attempts are recorded too. A notebook that includes
failures is more credible than one that lists only successes.

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
| A | Citrus Leaves Dataset (Rauf et al., 2019, Mendeley) | <https://github.com/ai-agriculture-circuits-and-systems/citrus_leaves> | Published 2019 | 420 (healthy 58 / canker 163 / greening 199) | |
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

### Experiment #2

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

## Checklist: required in the paper

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
