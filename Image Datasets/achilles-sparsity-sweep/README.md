# Achilles bag-size sparsity sweep

This separate workflow retains all 1–3 doctor-selected images from each source exam and fills the bag with images from other patients in the healthy group. Classes **0 and 1 are grouped as healthy**, and **2 and 3 as unhealthy**. Thus the healthy donor pool includes mild cases; “healthy” is the experiment's binary grouping, not a claim that class 1 has no pathology.

Compare bag sizes **10, 20, and 50** with average, max, ABMIL (dot-product), ABMIL (MLP), and transformer pooling. Selected source-image proportions are approximately 10–30%, 5–15%, and 2–6%, respectively. These are selected-image proportions, not verified pathology fractions. Healthy bags stay entirely within the healthy group; unhealthy bags have their source evidence diluted.

## Design and defaults

- Ten patient-level folds, stratified by the binary label. Each patient is validated once, with no separate test set.
- All selected images retained; patients with no selected images excluded and listed in `preparation_report.json`. The supplied metadata yields 85 eligible patients: 61 healthy-group and 24 unhealthy-group patients. This is verified again during preparation.
- Donors come from the source bag's training or validation partition, excluding the source patient. Sampling is without replacement **within each bag**; donor images may recur across bags. No cross-partition borrowing or automatic duplication is allowed.
- One donor sequence per source/fold, with nested subsets across bag sizes. All methods receive identical bag manifests; membership is fixed across epochs.
- Frozen ImageNet ResNet-50, randomly initialized classifier, ten epochs, last-epoch evaluation, no saved model weights. Frozen encoder BatchNorm statistics remain fixed.
- RGB-repeated grayscale, resize to 224, ImageNet normalization, no augmentation, matching the latest MNIST preprocessing. This does not inherit the medical sweep's RandAugment settings.
- ABMIL (MLP) uses one head. Dot-product attention and the two-layer transformer use 16 heads.
- Adam (learning rate 0.001, weight decay 0.001), StepLR (step 8, gamma 0.1), one bag per batch, matching the latest MNIST settings.

The metadata is read from `achilles_multiclass_metadata.csv` at the dataset root. For example, patient1's image number 2 maps to `patient1/img2.png`. Zero is a valid image number. Every selected image must exist; duplicate numbers, unexpected PNG filenames, and missing patient directories cause a clear error.

## Run on Palmetto

From `Image Datasets/achilles-sparsity-sweep`:

```bash
module load anaconda3/2023.09-0
source activate snakemake-slurm

snakemake --profile ../palmetto-sweep/profiles/palmetto --dry-run
snakemake --profile ../palmetto-sweep/profiles/palmetto
```

The profile supplies your existing partition/account/GPU settings. The configured `pytorch` Python executes preparation and training. Preparation runs locally on the controller and caches ImageNet weights; it does not train a model. **A dry run alone does not check donor capacity.** To check capacity before submitting GPU jobs, run:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto prepare
```

If a validation fold cannot supply enough distinct donor images for a 50-image bag, preparation writes the source/fold shortages and stops before training. Read `preparation_report.json`, then choose smaller bag sizes or explicitly revisit the sampling design. The roughly 800-image total does not guarantee sufficient donors within each held-out fold.

The default sweep submits **15 GPU jobs**, each training ten folds (150 model fits). To smoke-test all methods for one epoch, use a separate run name:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --config run_name=achilles_sparsity_smoke num_epochs=1
```

Keep settings unchanged while a controller is active; use a new run name when changing the experiment. Retries restart the affected ten-fold job from the beginning, rather than resuming saved weights. Other experiment folders and workflows are not modified.

## Outputs and interpretation

Results default to `../results/achilles_sparsity_frozen_v1/`:

| File | Contents |
|---|---|
| `design.json` | Fold assignments, image inventory, source/donor membership of every bag |
| `preparation_report.json` | Eligible counts, excluded patients, any donor shortages |
| `summary.csv` | One row per method/bag size; mean fold metrics with 90% CIs and pooled out-of-fold metrics |
| `fold_metrics.csv` | AUROC, sensitivity, specificity, balanced accuracy, Brier score, losses, confusion counts, severity subgroup metrics |
| `predictions.csv` | One held-out prediction per patient per method/bag size, including source count and original severity |
| `runs/nSIZE/METHOD/foldFOLD/history.csv` | Training and validation metrics per epoch |
| `logs/nSIZE/METHOD.log` | Job prints and errors |

Sensitivity and specificity use a predefined probability threshold of 0.5. Thresholds and epochs are not selected using held-out results. Class 2/3 subgroup sensitivity means detecting the unhealthy group, not correctly predicting severity; class 0/1 subgroup specificity means detecting the healthy group. Missing subgroups in a fold have blank metrics and zero counts.

AUROC, sensitivity, specificity, balanced accuracy, and Brier score receive two-sided 90% Student-t intervals across ten folds (9 degrees of freedom). These are approximate fold summaries: training sets overlap, validation folds are small, and bags share donors. Pooled out-of-fold metrics are reported separately and are not the same as mean fold metrics. Intervals are not clipped to [0, 1]. Do not treat the synthetic bags as extra independent patients or these CIs as independent-test guarantees.

Suggested figures plot AUROC, sensitivity, and specificity against bag size, with one curve per method. Sensitivity describes detection under dilution; specificity shows behavior on entirely healthy-group bags.
