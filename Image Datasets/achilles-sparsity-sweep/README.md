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
- ABMIL (MLP), dot-product attention, and the two-layer transformer use 16 heads.
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

## Extension: bag sizes 50 and 100

Use `sweep_large_bags.yaml` for **only the new 50- and 100-image conditions**.
It writes to `../results/achilles_sparsity_frozen_large_bags_v1/`, leaving the
previous small-bag run untouched. This avoids invalidating its design hash and
retraining completed small-bag conditions. The five methods require **10 GPU jobs**,
each training ten folds. Training settings and the fold seed are unchanged.

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_large_bags.yaml --dry-run
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_large_bags.yaml
```

For each source exam, use all distinct eligible healthy donor images in a
reproducible shuffled sequence first. If more images are required, sample the
remaining positions uniformly **with replacement** from that same donor pool.
An image may appear more than twice. Only distractors can be repeated: all 1–3
selected source images appear exactly once. Both bag classes use the same policy.
Donors remain in the same training/validation partition, and the source patient
is always excluded. An empty donor pool still stops preparation.

A separate deterministic replacement sequence preserves nesting: the 50-image bag
is a multiset subset of the 100-image bag. Smaller bag memberships are reproduced
exactly if metadata, image inventory, and seed match. Membership remains fixed
across epochs and identical across methods. This is replacement within a bag,
**not repeated training passes** through the dataset.

`bag_composition.csv` reports source count/fraction, distinct distractors, repeated
distractor occurrences, and total unique images for each bag and split. Inspect
this alongside `preparation_report.json`: validation donor pools are smaller, so
validation bags may repeat more images than training bags. Nominal source-image
fractions are 2–6% at size 50 and 1–3% at size 100; these do not represent verified
pathology fractions or 50/100 independent images. Increasing bag size can now
change both source proportion and duplicate frequency, which should be reported.

Download the new summary, fold metrics, predictions, and bag-composition report
alongside the old small-bag results to combine them for figures. Do not copy old
completion markers into the new run or overwrite the old design file.

For DTFD-AFS, CLAM-SB, and ProMIL, a matching extension configuration is provided
in `comparison-sweep/sweep_achilles_large_bags.yaml`. After preparation has created
the new Achilles `design.json`, run from `Image Datasets/comparison-sweep`:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_achilles_large_bags.yaml --dry-run
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_achilles_large_bags.yaml
```

This reuses the exact new bags, including repeated image occurrences, and adds
six comparator training jobs plus one feature-extraction job. Comparator results
go to `comparison_achilles_large_bags_v1`. The existing comparator schedules are
preserved; the shortened ProMIL schedule in the sample-efficiency experiment is
not applied here.

## Instance and standard-assumption baselines

`sweep_baselines.yaml` runs `instance` and `stan` at bag sizes 5, 10, 25, 50, and 100.
The instance baseline trains every image occurrence using its source bag label,
including healthy distractors in unhealthy bags, and averages image probabilities
for bag evaluation. The SA baseline (`stan`) trains on bag labels using noisy-OR.
Repeated donor images count repeatedly in both objectives, just as in the saved
bags. No additional repetition is introduced during training.

Both use a frozen ImageNet encoder and randomly initialized trainable classifier,
the same preprocessing and ten-epoch schedule as the pooling experiment, last-epoch
evaluation, and no saved models. Instance validation loss uses the averaged bag
probability; its training loss instead uses inherited image labels.

`source_runs` maps each bag size to an existing results folder. Defaults reuse
`achilles_sparsity_frozen_v1` for 5/10/25 and
`achilles_sparsity_frozen_large_bags_v1` for 50/100. Adjust these names if necessary.
Preparation imports the exact image lists, order, duplicates, and patient folds
from their `design.json` files, checks partition isolation, and rejects incompatible
source designs. It does not resample donors. Results are saved separately under
`results/achilles_sparsity_baselines_v1`, with the usual summary, fold metrics,
predictions, bag composition, and source-design hashes.

From this directory on Palmetto:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_baselines.yaml --dry-run
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_baselines.yaml --rerun-incomplete
```

There are ten GPU training jobs (5 sizes x 2 baselines), ten folds per job, with
32 GB memory and six hours per job. Preparation now runs as a CPU Slurm job;
summarization remains local. Existing completed results are not overwritten.
