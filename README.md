# X-ray baggage threat classification (DvXray, leak-free rebuild)

Binary image classification: does a baggage X-ray scan contain at least one annotated
prohibited item (**dangerous**) or not (**safe**)? Pretrained ViT-B/16, two training recipes,
evaluated once on a held-out test set that shares no bag, image or near-duplicate with training.

This is a rebuild of an older notebook (`AI_for_Airport_security.ipynb`, removed in this commit;
see git history), whose problems are
listed at the end. **Every number below was produced by the code in this repo; the run
logs and per-image predictions are in `results/`.**

> **Scale caveat, read first.** The machine used has no CUDA GPU (CPU-only, ~1.2 train img/s,
> 13.8 GB RAM with ~8 GB taken by another program). The models were therefore trained on a
> stratified, pair-preserving **CPU subsample: 400 train / 200 val / 800 test images**, 4 epochs
> each (~47 min total). The full split (22,389 / 4,800 / 4,798) is built and saved; run it on a
> GPU with `colab_full_run.ipynb`. Treat the results here as a small-sample study, not as the
> performance of the full-data model.

## Data

- **Source:** DvXray, as redistributed in the Hugging Face dataset
  [`FrenchCastle/xray-baggages-customs`](https://huggingface.co/datasets/FrenchCastle/xray-baggages-customs)
  (files `annotations/dvxray_coco.json`, `images_dvxray.zip`, `MANIFEST.csv`; 7.2 GB).
  Upstream: https://github.com/Mbwslib/DvXray. Ma et al., *IEEE TIFS* 19, 2024.
- **Cross-scanner set (optional step):** `sixray_yolo_subset` from the same repository (909 MB),
  a community YOLO re-annotation of SIXray (Miao et al., CVPR 2019).
- 32,000 images = 16,000 bags × 2 orthogonal views (`OL`, `SD`), 800×600 px, one scanner.

### Licence (recorded from the dataset card, 2026-10-14)

- The card's licence is `other` / "per-source research terms". There is no single licence. Each
  image keeps its source's terms; the HF maintainer releases only the annotation conversion and
  label mapping under CC BY 4.0.
- **DvXray: "Academic use only."** **SIXray subset: "Academic use only (SIXray terms)."**
- The card is gated. Its access prompt asks users to use the data for **non-commercial research
  only**, to follow each source's terms, to **cite the original authors**, and **not to use the
  data to develop methods for evading security screening**. Anything beyond non-commercial
  research needs the source authors' permission.
- Redistribution is not permitted, so **no images are in this repo**. `xray_baggage/` is in
  `.gitignore`; download the data yourself (below).

Citation for DvXray: Ma, Jia, Li, Wu, Wang, Chen, "Toward Dual-View X-Ray Baggage Inspection: A
Large-Scale Benchmark and Adaptive Hierarchical Cross Refinement for Prohibited Item Discovery,"
*IEEE Transactions on Information Forensics and Security* 19, 2024.

### Labels and class balance

Label = dangerous if the image has ≥ 1 bounding box in the COCO file, else safe.

| | images |
|---|---:|
| safe | 22,400 |
| dangerous | 9,600 |

Threat boxes by original DvXray class: Lighter 767, Saw_blade 753, Hammer 749, Wrench 729,
Knife 727, Battery 726, Gun 723, Bat 714, Pliers 712, Pressure_vessel 699, Screwdriver 690,
Scissors 690, Dart 660, Fireworks 658, Razor_blade 586 (10,583 boxes). The full breakdown, including unified
class names, is in `results/data_report.json`.

**Label caveat:** 400 images in DvXray's `positive/` folder have no box. Under the box rule
they are labelled *safe*, even though their paired view contains a threat. They were kept as
specified. The test metrics are also reported with those images excluded (9 of them fall in the
800-image test subsample); see `sensitivity_excluding_unboxed_positive_folder_images` in
`metrics.json`.

### Split method (`prepare_data.py`)

1. **Pairs.** File names are `positive|negative/<P|N>#####_<OL|SD>.png`; the two views of a bag
   share the prefix. The code asserts every bag has exactly one `OL` and one `SD` view (16,000/16,000).
2. **Official split rejected.** In `MANIFEST.csv`, **5,391 of 16,000 bags have their two views
   in different splits**, so the official split would leak. A new split was built instead.
3. **Perceptual-hash dedup.** First attempt: the default 64-bit pHash of the full frame. It
   collapsed: 77,144 image pairs had *identical* hashes, and 11,407 bags merged into one
   "duplicate" group. On inspection these were different bags; the mostly white background
   dominates the hash. Final method: crop each image to its non-background bounding box (grey
   < 235), then a **256-bit pHash** (`hash_size=16`). The resulting distance histogram has a clear
   gap (4 pairs at 0, then 26, 32, 36…). Inspecting every pair ≤ 48 bits showed:
   - d = 0: exact copies (bags P04244 = P04443 and P04245 = P04251, both views);
   - d = 26–44: the same bag re-scanned with near-identical contents;
   - d ≥ 46: the same suitcase *shell* packed with different contents.

   **Threshold = 44 bits.** Bags linked by a duplicate are grouped (union-find) so duplicates
   never cross splits, then one image per duplicate cluster is kept: **13 duplicates removed**
   (11 train, 2 test). The code **asserts zero cross-split overlap** by file name, bag, and
   pHash distance ≤ 44.
4. **Stratified, pair-grouped 70/15/15** by group label (seed 42):

   | split | images | bags | dangerous | safe |
   |---|---:|---:|---:|---:|
   | train | 22,389 | 11,198 | 6,711 | 15,678 |
   | val | 4,800 | 2,400 | 1,441 | 3,359 |
   | test | 4,798 | 2,399 | 1,442 | 3,356 |

5. **CPU subsample** (whole bags, stratified, drawn *inside* each split so it inherits the
   guarantees): train 400 (120 dangerous), val 200 (60), test 800 (241), 400 bags.

The split CSVs are in `splits/` (`train/val/test.csv`, `sub_*.csv`). Each row holds file name,
bag, view, label, box count, threat classes, and pHash.

## Models and training (`train.py`)

| | baseline | tuned (old-notebook recipe) |
|---|---|---|
| backbone | torchvision ViT-B/16, IMAGENET1K_V1, all layers trained | same |
| head | Linear(768→2) | Dropout 0.5 → Linear 256 → ReLU → Dropout 0.4 → Linear 2 |
| augmentation | none | h-flip, rotation 10°, ColorJitter(0.2, 0.2, 0.2, hue 0.1) |
| optimiser | Adam, lr 1e-4 | AdamW, lr 4e-4, weight decay 2e-4 |
| schedule | none | StepLR(step 3, γ 0.6) |
| mixed precision | no | yes: fp16 + GradScaler on CUDA, **bf16 autocast on CPU** (see notes) |
| grad clipping | no | max-norm 0.8 (applied after `unscale_`) |

Shared settings: seed 42 everywhere (Python, NumPy, torch, DataLoader generator, deterministic
algorithms), the same split, 4 epochs, batch size 8, input resized to 224×224 (whole bag, no
crop), ImageNet normalisation. **Checkpoint = epoch with the best validation AUROC**; validation
probabilities at that epoch are saved for threshold selection. The test split is never read in
`train.py`.

**Class imbalance: inverse-frequency weighted cross-entropy** (weights safe 0.714, dangerous 1.667
on the subsample). I chose this over a weighted sampler because, at 400 training images, a sampler
would repeat the 120 positives about 2.3× per epoch and drop many negatives. It would also make
the effective data differ between runs. A weighted loss sees every image exactly once per epoch, so
both models get the same number of steps on the same data.

Notes: the baseline learning rate (1e-4) was not specified in the brief; I took it from the old
notebook's first optimiser. CPU autocast supports only bf16, whose exponent range matches fp32,
so loss scaling is unnecessary there and the GradScaler is disabled on CPU. On a GPU the
specified fp16 + GradScaler path runs.

## Results: DvXray test subsample (800 images, 400 bags, 241 dangerous)

The test set was run **once** (`evaluate.py`; predictions saved in `results/test_preds_*_sub.csv`,
and the script refuses to re-run the models when those files exist). Decision threshold 0.5.
95% CIs come from a **bag-level cluster bootstrap** (2,000 resamples; both views of a bag
resampled together, because they are correlated).

| metric | baseline | tuned |
|---|---|---|
| accuracy | **0.753** [0.719, 0.785] | 0.699 [0.655, 0.744] |
| threat recall | **0.909** [0.869, 0.947] | 0.000 [0.000, 0.000] |
| precision | 0.554 [0.492, 0.614] | undefined (no positive predictions; reported as 0) |
| F1 | 0.689 [0.639, 0.736] | 0.000 |
| AUROC | **0.891** [0.865, 0.914] | 0.725 [0.678, 0.769] |
| false-alarm rate | 0.315 | 0.000 |
| confusion (TN/FP/FN/TP) | 383 / 176 / 22 / 219 | 559 / 0 / 241 / 0 |
| best epoch (val AUROC) | 4 (0.886) | 3 (0.676) |

Paired bootstrap, tuned − baseline: AUROC −0.166 [−0.215, −0.118]; threat recall −0.909
[−0.947, −0.869]; accuracy −0.054 [−0.115, +0.008].

**The tuned recipe failed in this setting.** Its training loss stayed near 0.70–0.76 for all four
epochs, and every test probability fell between 0.330 and 0.477, so nothing crossed 0.5. Its 0.699
accuracy equals the fraction of safe images. I didn't test the cause. The likely suspects are the
4e-4 learning rate on a fully unfrozen ViT, the hue jitter (X-ray pseudo-colour encodes material),
and the dropout head, all with only 400 images. No extra tuning was done after seeing the test set.

Confusion matrices: `results/confusion_matrix_sub.png`; ROC curves: `results/roc_sub.png`.

### Threshold analysis: false-alarm rate needed for a target threat recall

| target recall | model | test-ROC operating point (optimistic: threshold picked on test) | threshold picked on val, applied to test |
|---|---|---|---|
| 90% | baseline | FAR **0.256** (recall 0.900) | thr 0.364 → recall 0.925, FAR 0.352 |
| 95% | baseline | FAR **0.415** (recall 0.950) | thr 0.274 → recall 0.946, FAR 0.392 |
| 90% | tuned | FAR 0.644 (recall 0.900) | thr 0.333 → recall 0.954, FAR 0.760 |
| 95% | tuned | FAR 0.760 (recall 0.954) | thr 0.333 → recall 0.988, FAR 0.809 |

The right-hand column is the honest deployment estimate: the threshold is fixed before the test
set is seen. It rests on only 60 dangerous validation images, so it is noisy.

### Sensitivity: excluding the 9 unboxed `positive/` images from the test set

Baseline: accuracy 0.761, recall 0.909, AUROC 0.898, FAR 0.304. Tuned: accuracy 0.695, AUROC 0.731.

## Shortcut-learning checks (no test data used)

DvXray's positive and negative bags were collected as separate sets, so both checks below use
train/validation images only.

1. **Global-statistics probe** (`shortcut_probe.py`): logistic regression on a 3×32-bin colour
   histogram plus foreground fraction, mean and std. No spatial or object information.

   | | val AUROC (subsample: 400 train → 200 val) | val AUROC (full: 22,389 → 4,800) |
   |---|---:|---:|
   | histogram logistic regression | 0.772 | **0.883** |
   | foreground fraction alone | 0.736 | **0.769** |

   Global colour statistics alone almost match the ViT's 0.891 test AUROC. This doesn't prove a
   shortcut, since dense metal objects legitimately shift the colour histogram. But how much of the
   bag area is non-background predicts the label at AUROC 0.77, which suggests dangerous bags
   systematically differ in size or packing density.

2. **Box-occlusion test** (`occlusion_test.py`): 300 dangerous validation images not used for
   checkpoint selection. Each is scored three ways: original; every threat box painted white; and a
   control with the same boxes shifted to a random spot on the bag.

   | baseline (subsample checkpoint) | mean P(dangerous) | flagged at 0.5 |
   |---|---:|---:|
   | original | 0.873 | 0.893 |
   | threat boxes masked | 0.828 | **0.857** |
   | control: shifted masks | 0.869 | 0.890 |

   Masking the threat lowers the flag rate by only 0.037 [95% CI 0.003, 0.070] (paired
   bootstrap); the control lowers it by 0.003 [−0.010, 0.017]. **With the threat removed, the
   baseline still flags 86% of these bags.** At this training scale the classifier relies mostly on
   something other than the annotated object. Whether this persists with full-data training is
   untested.

## Cross-scanner test: SIXray, no retraining (`cross_scanner.py`)

829 SIXray-subset *test* images with ≥ 1 box (Gun 288, Pliers 387, Wrench 214, Knife 111,
Scissors 97). The subset has almost no negatives, so only recall is reported. Image-level
bootstrap CIs.

| model | recall @ 0.5 | recall @ DvXray-val 90% thr | recall @ DvXray-val 95% thr |
|---|---|---|---|
| baseline | 0.996 [0.992, 1.000] | 0.999 [0.996, 1.000] | 0.999 [0.996, 1.000] |
| tuned | 0.001 [0.000, 0.004] | 1.000 | 1.000 |

**Do not read this as cross-scanner robustness.** A diagnostic (`sixray_unboxed_diagnostic.py`)
scored the subset's 119 images that have *no* boxes (115 of them from the train split): the
baseline flags **84%** of them at 0.5, and the tuned model flags 100% at its validation
thresholds. These images are not guaranteed threat-free, so this is not a valid false-alarm rate.
Still, it strongly suggests the near-perfect recall comes from flagging almost any image from the
new scanner. Per-class recall is in `metrics.json`.

## Reproduce

```bash
pip install -r requirements.txt
hf download FrenchCastle/xray-baggages-customs --repo-type dataset --local-dir xray_baggage --include "annotations/dvxray_coco.json" "images_dvxray.zip" MANIFEST.csv
```
Unzip `images_dvxray.zip` inside `xray_baggage/` (and the SIXray zip, for the optional step).
Then:

```bash
python prepare_data.py
python train.py --model baseline
python train.py --model tuned
python evaluate.py
python cross_scanner.py
python sixray_unboxed_diagnostic.py
python shortcut_probe.py --full
python occlusion_test.py
```

Add `--full` (and `--batch-size 32 --workers 2` on a GPU) for the full split, or use
`colab_full_run.ipynb` on a Colab T4. All outputs go to `results/` (`metrics.json` collects
everything); checkpoints go to `checkpoints/` (git-ignored).

## Limitations

- **Small sample.** 400 training images and 4 epochs on CPU. The CIs reflect test-set sampling
  only, not training variance; there is a single seed and a single run per model.
- **Shortcut evidence.** The global-statistics probe and the occlusion test both indicate the
  classifier uses features other than the threat object. High AUROC here isn't evidence the model
  "sees" weapons.
- **Suitcase reuse.** The DvXray authors reuse suitcase shells across bags (pairs at pHash
  distance ≥ 46 are the same shell with different contents). These were *not* treated as
  duplicates, so the same luggage can appear in train and test.
- **Label noise.** The image-level box rule labels 400 positive-bag views as safe. Bag-level (pair)
  classification would avoid this but was out of scope.
- **Tuned recipe untuned.** It was run exactly as specified and failed; this says nothing about
  augmentation or regularisation in general.
- **Mixed precision on CPU** used bf16 rather than the fp16 + GradScaler used on GPU.
- **Cross-scanner.** SIXray has almost no negatives, so recall alone is uninformative (see above).
  The SIXray subset is a community re-split, resized to ≤ 640 px, not the official release.
- **One scanner, staged bags.** DvXray is a single-scanner benchmark; nothing here says anything
  about operational screening.

## What the old notebook got wrong

- `val/` was a copy of `train/`, the validation loader was never used, and the reported "97.21%"
  was the *mean training accuracy across epochs*.
- No test set and no evaluation code.
- Dangerous images (e.g. `B0001_0007..jpg`) and safe images (`grayscaled34.jpg`) came from
  different sources, so the classes were separable by source alone.
- The bounding-box head regressed a constant target ([0.3, 0.3, 0.7, 0.7] for every dangerous
  image), and the drawn box was hard-coded. That model (`ViTWithDetection`) was defined but never
  trained; a different torchvision model was.
- Gradient clipping was applied to *scaled* gradients (no `scaler.unscale_` before
  `clip_grad_norm_`), so the 0.8 threshold did not mean what it says.
- Normalisation mean/std of 0.5 instead of the ImageNet statistics the weights expect.
- 359 training images, batch size 6, no seeds, no class-balance handling.
