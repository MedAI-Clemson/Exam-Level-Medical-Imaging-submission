# Paper-method comparisons

Independent Snakemake/Slurm comparisons for **DTFD-MIL (AFS)**, **CLAM-SB**, and **ProMIL**. The existing medical, MNIST, and Achilles sparsity workflows are unchanged. Methods have their own encoders and training schedules rather than a shared pretraining/freezing grid. See [SOURCES.md](SOURCES.md) for the author repositories, reference commits, licenses, and exact adaptations.

## What runs

| Experiment | Conditions | Training jobs | Fits |
|---|---:|---:|---:|
| MNIST | 12 existing bag-size/density pairs | 36 | 36 |
| Achilles sparsity | Bag sizes 5, 10, 25 | 9 | 90 |
| Original medical datasets | Achilles, CT, MRI, X-ray | 12 | 120 |
| **All enabled** | **19** | **57** | **246** |

Six additional GPU jobs extract frozen image features, once per dataset. CLAM and DTFD reuse these caches across all relevant conditions and folds. Features are stored as float32, roughly 4096 bytes per unique image, and loaded through memory-mapped arrays. No trained model files are saved; feature caches are not model checkpoints.

Select experiments in `sweep.yaml`, for example:

```yaml
experiments: [achilles_sparsity]
methods: [dtfd_afs, clam, promil]
```

This subset uses nine training jobs plus one feature-extraction job. It is a reasonable first run after the smoke test. `num_heads` is not a setting for these methods and does not change your existing 16-head attention baselines.

## Data and previous runs

- `mnist_source_run: mnist_composition_v2` reads exact saved image-index lists from that run's `designs/nSIZE/pDENSITY.json`, with support for the legacy `design.json`. It preserves train/validation/test membership and target counts.
- `achilles_source_run: achilles_sparsity_frozen_v1` reads exact folds and bags from its `design.json`. Classes 0/1 remain healthy-group and classes 2/3 unhealthy-group. No donor resampling occurs. The selected sizes must already exist in the saved design; current defaults match the supplied 5/10/25 summary.
- Original medical datasets reuse `../config/*_config.yaml` for paths, labels, filtering, and intensity preprocessing, and the existing patient-fold generator with seed 123. They do not inherit model/pretraining selectors. Verify these data configs still describe the runs you want to compare; this workflow does not infer historical patient splits from old result tables.

Preparation fails on missing source designs, selected bag sizes, images, or partition overlap. It does not silently regenerate different sparsity bags. Paths in old Achilles manifests must still exist on Palmetto. `project_dir`, `results_dir`, and `mnist_data_dir` resolve relative to this workflow when not absolute. Keep the original source result folders available.

## Run on Palmetto

From `Image Datasets/comparison-sweep`:

```bash
module load anaconda3/2023.09-0
source activate snakemake-slurm
snakemake --profile ../palmetto-sweep/profiles/palmetto --dry-run
snakemake --profile ../palmetto-sweep/profiles/palmetto prepare
```

The existing profile supplies your account, partition, and GPU selection. Preparation runs on the controller; feature extraction and training run as GPU jobs using the configured `pytorch` Python. Existing packages are sufficient: PyTorch, torchvision, timm, NumPy, pandas, scikit-learn, SciPy, PyYAML, Pillow, and pydicom. No new third-party model package is required.

Run the CPU model checks in the training environment before submitting:

```bash
/home/bformby/.conda/envs/pytorch/bin/python -m unittest discover -p 'test_*.py' -v
```

For a one-epoch smoke test, use a separate result name. The override applies to every method without changing their full-run defaults:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --config run_name=paper_methods_smoke epochs_override=1
```

Then launch the full configured sweep:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto
```

Defaults are much longer than the prior 10-epoch experiments: 200 epochs for CLAM/DTFD, up to 100 for MNIST ProMIL, and 125 for medical ProMIL. Each medical/Achilles job contains ten folds. Calibrate `runtime_minutes` and memory after the smoke test. A failed job restarts its folds from the beginning; there are no saved trained weights to resume.

## Results

Default output: `../results/paper_methods_v1/`.

- `manifest.json`: exact bag memberships, folds, image inventory, and source-design hashes.
- `features/DATASET/`: shared frozen feature matrix and completion receipt.
- `runs/CONDITION/METHOD/`: resolved settings, per-fold history and predictions, combined predictions, fold metrics, summary JSON.
- `summary.csv`: means and 90% fold-based t intervals for AUROC, sensitivity, specificity, balanced accuracy, and Brier score, plus pooled metrics.
- `fold_metrics.csv` and `predictions.csv`: all conditions combined, including IDs and class probabilities.
- `logs/CONDITION/METHOD.log`: training prints and errors.

MNIST has one test evaluation per configuration and therefore no fold-based CI. Medical/Achilles report ten held-out folds with approximate CIs; folds share training patients and synthetic bags may share donors. Binary sensitivity/specificity use threshold 0.5. Multiclass AUROC is macro one-vs-rest, sensitivity/specificity are macro class averages, and multiclass Brier is the mean sum of squared class-probability errors. Missing-class AUROC is blank, not zero. ProMIL X-ray results must be labeled OVR as explained in SOURCES.md.

These runs evaluate complete methods with different native encoders, objectives, and training budgets. Do not interpret differences as pooling-only effects. Medical preprocessing and label definitions stay fixed, but architecture-specific image resizing and normalization follow the documented profiles.

Keep settings unchanged while jobs are active and use a new run name for a changed experiment. Existing results are never relabeled or overwritten by this comparison sweep.

## Local verification status

The standard-library design tests and Python syntax checks were run during implementation. Model forward/backward tests are included, but require PyTorch, which was unavailable in the local editing environment. Snakemake/Slurm and actual image training must be verified on Palmetto; run the test command and one-epoch smoke sweep above before committing to the full run.
