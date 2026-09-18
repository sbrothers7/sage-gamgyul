# Data Collection Guide

The study uses two kinds of data. The point of the study is that they differ.

| | Folder | Purpose | Source |
|---|---|---|---|
| A | `source_A` | Reproduce the original paper | The public dataset used by the paper |
| B | `source_B` | **Generalization test** | Citrus images from a different source |

The study only holds if B differs enough from A. Splitting one dataset in two
is just a train/test split, so B images have to come from a **different
source**.

---

## Folder structure

Folder names are the labels, so this structure is required.

```
data/raw/
├── source_A/
│   ├── healthy/      ← healthy leaves
│   │   ├── 0001.jpg
│   │   └── ...
│   ├── canker/       ← citrus canker
│   └── greening/     ← huanglongbing (HLB)
└── source_B/
    ├── healthy/
    ├── canker/
    └── greening/
```

- Folder names match `classes` in `config.yaml` exactly, including case:
  `Healthy` and `healthy` are different.
- jpg, png, bmp and webp all work.
- Subfolders are allowed and are found automatically. In `source_B`, one
  subfolder per origin (`source_B/<class>/<origin>/`) adds a per-origin
  breakdown to stage 3.

---

## A: reproduction data

The paper being reproduced is **Devora-Guadarrama et al. (2025)**, which used a
public citrus dataset from Kaggle.

### Where to find it

- Kaggle (free account, <https://www.kaggle.com>). Useful search terms:
  - `citrus leaf disease`
  - `citrus canker huanglongbing`
  - `citrus plant disease dataset`
- The downloaded images go into the folder structure above, one folder per
  class.

### What to check

Class names differ between datasets. HLB, for example, appears under several
names:

- `greening` / `HLB` / `Huanglongbing` / `citrus_greening`

Any name works, as long as the folder names and `config.yaml` agree.

### Number of images

At least 50 per class, ideally 100 or more. With fewer, the model doesn't
train properly and the paper's numbers can't be reproduced.

---

## B: new citrus images for the generalization test

This is the core of the study. B images need to come from **conditions
different from A**.

### Option 1 — own photos (best)

Photos taken at a Jeju citrus farm or in a school greenhouse.

- One leaf per photo, filling most of the frame.
- A smartphone is enough, and it is closer to how the model would be used.
- Shooting conditions deliberately varied: sunny, cloudy, morning, afternoon,
  shade and direct sun. This variety is what makes B different from A.
- Background left uncontrolled. Soil, other leaves or sky in the frame are
  fine, and more realistic.

**Every diagnosis needs to be confirmed by an expert.** A photo judged "HLB" by
eye is not a valid label, and wrong labels invalidate the whole study.
Possible contacts:

- Jeju Special Self-Governing Province Agricultural Research and Extension
  Services (citrus)
- National Institute of Horticultural and Herbal Science, Citrus Research
  Institute
- Local agricultural technology centers

The research notebook records who confirmed which photos, and when. This
record makes the paper much more credible.

### Option 2 — another public dataset

When taking photos isn't possible, B can be a public dataset from a
**different source** than A:

- PlantVillage (Kaggle / GitHub): citrus coverage needs checking
- Mendeley Data: search `citrus disease`
- Roboflow Universe: search `citrus`
- Datasets published as paper supplements

**The same dataset is never split between A and B.**

### Download tools used so far

Public datasets are usually hosted on GitHub, Kaggle or Hugging Face, and each
is downloaded differently.

**GitHub repository (git sparse-checkout):** downloads only the folders that
are needed, not the whole repository.

```bash
git clone --depth 1 --filter=blob:none --sparse <repo-url>
cd <repo>
git sparse-checkout set <folder-path>
```

**Kaggle (`kagglehub`):** public datasets download without logging in.

```bash
pip install kagglehub
python -c "
import kagglehub
path = kagglehub.dataset_download('owner/dataset-name')
print(path)
"
```

Private and competition datasets need `~/.kaggle/kaggle.json`, which is issued
in the Kaggle account settings.

**Hugging Face Hub:** some datasets need an access token.

```bash
pip install -U huggingface_hub
hf auth login   # paste the token
```

The token is stored in `~/.cache/huggingface/token`. **It must never be written
into `config.yaml` or the code.** A committed token is exposed to anyone with
access to the repository.

> Downloads are kept unchanged in `data/datasets/<dataset name>/`. Only the
> needed class folders are copied into `data/raw/source_A` or `source_B`. This
> keeps each image's origin traceable and makes mistakes easy to undo.
> `data/datasets/` is in `.gitignore`, so it never reaches the repository.

### Number of images

20–30 per class is enough, because B is only evaluated, never trained on.
Below 15 per class, results depend too much on chance.

---

## After adding data

```
python run_1_prepare.py
```

This command checks:

1. that each class has enough images
2. that the folder names match the configuration
3. **whether any image appears in both A and B** (the most important check),
   and whether either source contains exact duplicates of its own images

Any duplicate found by check 3 has to be removed. If the same photo is in both
A and B, the claim "tested on images not used for training" is false. This is
**data leakage**, which Kapoor & Narayanan (2023) found in 294 papers across
17 fields. The check only catches byte-identical copies. Resized or re-saved
copies need a perceptual-hash check (see the cleaning steps below).

---

## Data provenance

This table is required in the paper.

| | Source | URL or location | Published / taken | Images | Labels confirmed by |
|---|---|---|---|---|---|
| A | Citrus Leaves Dataset (Rauf et al., 2019, Mendeley) | <https://github.com/ai-agriculture-circuits-and-systems/citrus_leaves> | Published 2019 | 420 (healthy 58 / canker 163 / greening 199) | (not yet confirmed by an expert) |
| B-1 | Plant Village Orange — Huanglongbing | <https://github.com/ai-agriculture-circuits-and-systems/Plant_Village_Orange> | PlantVillage release | greening 5,507 | (not yet confirmed) |
| B-2 | kaku321 / citrus-plant-disease (Kaggle) | <https://www.kaggle.com/datasets/kaku321/citrus-plant-disease> | Unknown (appears web-scraped) | healthy 100 / canker 52 / greening 69 | (not yet confirmed) |
| B-3 | CitrusUAT (Gómez-Flores et al., 2024) | <https://zenodo.org/records/8294078> | Published 2024 | healthy 100 / greening (HLB) 43 | HLB confirmed by qPCR in the original paper |

`source_B` uses the layout `data/raw/source_B/<class>/<origin>/`, with origins
`PlantVillage`, `kaku321` and `CitrusUAT`, so results can be broken down by
origin.

`python build_data.py` downloads these datasets and applies the cleaning below
to produce `data/raw/`. This way the collaborator gets identical data.

**Cleaning steps (why the counts are lower than the originals).** These belong
in the paper's data section.

- **Plant Village Orange:** half of the 11,014 files are byte-identical copies
  (named `_1`). 5,507 were kept.
- **kaku321:** most files are augmentation blocks, meaning one photo flipped,
  rotated and zoomed into 48–56 files with consecutive numbers. Only the first
  image of each block (the original) was kept, reducing 9,766 files to 221
  photos.
  - In the greening folder some blocks are only partly present. Where the
    original is missing, the lowest-numbered image was kept.
  - One healthy photo appeared as two separate blocks, and the second copy was
    removed.
- **Citrus Leaves (A):** 5 greening images were re-saved or brightness-changed
  copies of other photos (nos. 9, 28, 45, 79 and 188). They were removed to
  prevent train/test leakage, reducing A from 425 to 420 images.
- **Between A and B:** checked with file hashes (MD5) and with a perceptual hash
  (dHash) that ignores flips and rotations. No overlap was found.

> B is very unbalanced: greening 5,619, healthy 200 and canker 52. Overall
> accuracy is dominated by greening, which is mostly PlantVillage, so results
> are reported both per class and per origin. Canker comes from one origin (52
> images), so its confidence interval is wide.

**Tried but not used:** Kaggle `dtrilsbeek/citrus-leaves-prepared`, downloaded
with `kagglehub.dataset_download`. Its class counts (healthy 58, canker 163,
greening 204) match A exactly, so it is a repackaged copy of the same source.
It was used neither as B nor as a replacement for A, and is kept only in
`data/datasets/citrus-leaves-prepared/`.

> Kamilaris & Prenafeta-Boldú (2018), in their review of agricultural
> engineering, treat data provenance and preprocessing as core reporting items,
> not side details.
