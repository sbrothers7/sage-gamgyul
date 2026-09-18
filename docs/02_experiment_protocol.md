# Experiment Protocol

What is done, in what order, and why.

---

## Research question

> When a published citrus disease model is reimplemented, does its reported
> performance hold on new citrus images that were not used for training?

The answer is a single number: **generalization gap = reproduction
performance − generalization performance**.

---

## Experiment design

The design follows the intra-dataset / cross-dataset comparison of Mahapatra
et al. (2026). One model is evaluated under four conditions.

| Condition | Data | Background | What it shows |
|---|---|---|---|
| 1 | source_A test set | original | **Reproduction**: whether the paper's numbers are reproduced |
| 2 | all of source_B | original | **Generalization**: whether the model works on new images |
| 3 | source_A test set | removed | Whether reproduction holds without the background |
| 4 | all of source_B | removed | Whether the background caused the drop |

Conditions 3 and 4 exist because the model may have learned the
**photographic setting (the background)** rather than the lesions. If
condition 1's accuracy drops sharply once the background is removed, the
model was identifying which dataset a photo came from, not diagnosing the
disease.

---

## Steps

`./run_all.sh` runs stages 1–5 in order. `./run_all.sh <model>` does the same
for a specific model.

### Stage 0 — pipeline check (no real data)

This stage confirms that the code runs before real data is available.

```
python make_test_data.py
python run_1_prepare.py
python run_2_train.py
python run_3_evaluate.py
python run_4_gradcam.py
python run_5_report.py
python make_test_data.py --clean
```

The numbers from this run are **not research results**. The images are
synthetic and only test the plumbing. `--clean` removes the fake data
afterwards.

### Stage 1 — data preparation

```
python run_1_prepare.py
```

- Builds the image lists and splits A into train, validation and test sets
  (70 / 15 / 15).
- Uses a **stratified split**, so class proportions are the same in every set.
- Checks for identical images between A and B, and for exact duplicates within
  each source. Duplicates have to be removed before continuing.
- Creates background-removed copies, clearing the previous ones first.

### Stage 2 — training (reproduction)

```
python run_2_train.py [--model <name>]
```

- Fine-tunes an ImageNet-pretrained model on the citrus classes (transfer
  learning).
- Saves the checkpoint with the best validation F1.
- Stops automatically after 7 epochs without improvement.

**Accuracy at this stage needs to be close to the paper's.** Much lower
accuracy means the data is insufficient or the settings are wrong. In that
case a stage 3 result would mean "the model never learned", not "the model
doesn't generalize".

### Stage 3 — generalization test ★ core ★

```
python run_3_evaluate.py [--model <name>]
```

Prints results for the four conditions and a final comparison table. When
`source_B` uses per-origin subfolders, it also prints a breakdown by origin.
**This output is the answer to the research question.** A screenshot of it
goes into the research notebook.

### Stage 4 — what the model looked at

```
python run_4_gradcam.py [--model <name>]
```

- Grad-CAM heatmaps show where the model looked in each image.
- The **leaf focus** score is close to 1 when the model looked at the leaf and
  close to 0 when it looked at the background.
- Comparing leaf focus between A and B can explain a drop in performance.

Leaf segmentation fails when the background is a similar color to the leaf.
Failed images are left out of the average automatically, and a warning is
shown. **If fewer than half of the samples are valid, the leaf-focus numbers
don't go in the paper.**

### Stage 5 — figures and tables

```
python run_5_report.py [--model <name>]
```

Four figures are written to `results/figures/`, or to `results/<model>/figures/`
when `--model` is used. `fig3_condition_comparison.png` is the paper's main
figure.

---

## Interpreting the results

`source_B` is unbalanced, so plain accuracy can be misleading: a model that
always predicts the largest class scores high. Balanced accuracy (macro
recall) and MCC are the reliable measures for B.

### Large gap (20 percentage points or more)

The hypothesis is supported. These narrow down the cause:

1. `fig4_per_class_drop.png`: which classes fail most
2. The confusion matrix: what is mistaken for what
3. Leaf focus: whether the model's attention drifts off the leaf
4. The background-removed conditions: whether removing the background shrinks
   the gap
5. The per-origin breakdown: whether the same class scores differently
   depending on where the images came from

If the gap shrinks in point 4, "background dependence is the cause" is a
well-supported conclusion.

### Small gap (under 5 percentage points)

A small gap alone does not show good generalization. These are checked first:

- **Is B really different from A?** If B comes from the same dataset or was
  taken under similar conditions, a small gap is expected.
- **Is B large enough?** With fewer than 15 images per class, the result may
  be chance.
- **Are there duplicates between A and B?** The stage 1 check answers this.

If all three hold and the gap is still small, that is a meaningful result in
itself. "Transfer-learning citrus disease models are more robust than
expected" is a valid conclusion for a paper.

### Negative gap (B scores higher than A)

This almost always signals a problem. The possible causes, checked in order,
are: B is easier than A, there are duplicates, or labels are wrong.

---

## Rules

**1. The seed stays fixed.**
`seed: 42` in `config.yaml` exists for reproducibility. Trying different seeds
and reporting only the best one is manipulation, not research. Running
several seeds is acceptable only when every seed's result is reported.

**2. source_B is never used for training.**
A single source_B image in training invalidates the study.

**3. The result is not decided in advance.**
"The model won't generalize" is a hypothesis, not a conclusion. An opposite
result is reported as it is.

**4. Every run is recorded.**
The date, settings and result of every run go into
`docs/03_research_notebook.md`.

---

## Changing experiment conditions

Only `config.yaml` changes. The code itself doesn't need editing.

| Goal | Setting |
|---|---|
| Use a different model | `model.name`, or `--model <name>` for a single run |
| Train longer | `train.epochs` |
| Add a class | `classes`, plus a matching folder |
| Training is slow | `train.batch_size` set to 8 |
| Out-of-memory error | `train.batch_size` set to 4–8 |

To compare several models, stages 2–5 are run once per model with
`--model <name>`, or with `./run_all.sh <name>`. Each model's results go to
`results/<name>/`, so runs don't overwrite each other.
