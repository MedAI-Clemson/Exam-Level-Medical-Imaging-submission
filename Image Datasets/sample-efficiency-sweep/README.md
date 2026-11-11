# Medical sample efficiency

This sweep varies the fraction of **training patients**: 10%, 20%, 50%, and 100%.
It uses Achilles, CT, MRI, and X-ray with the same patient-grouped, stratified
10-fold construction and split seed as the medical workflow. Validation patients
and their complete bags stay fixed at every fraction. Existing results are not
overwritten or reused as new training runs.

## Methods and training

- Average, max, ABMIL (dot-product), ABMIL (MLP), and transformer: **frozen
  ImageNet encoder, random trainable classifier**, with trainable pooling layers.
  Dataset YAML files still control the architecture, number of attention heads,
  dropout, RandAugment/preprocessing, batch size, optimizer, scheduler, and epoch
  budget. The sweep explicitly overrides instance/RadImageNet loading, encoder
  freezing, classifier freezing, checkpoint selection, and model saving.
- DTFD-AFS and CLAM-SB: reuse the existing frozen ImageNet ResNet-50 layer-3
  features and method-specific losses. One feature cache per dataset is shared
  across fractions and folds; the extractor is pretrained and never fitted on
  these patients. Features occupy disk space even though model saving is off.
- ProMIL: reuse its existing ImageNet ResNet-18 medical profile, with 5
  frozen epochs followed by 20 fine-tuning epochs. This
  comparator is **not frozen throughout**. X-ray uses the existing OVR extension.
  No method loads an instance-pretrained encoder or classifier.

Each epoch processes every selected training bag once, in shuffled order. Smaller
fractions therefore require fewer updates. The five pooling methods retain their
dataset batch sizes, including a smaller final batch. CLAM/ProMIL take one bag per
update; DTFD retains its two-tier objective. Learning-rate schedules remain epoch-based.
ProMIL uses parallel bag loading, pinned memory, and worker prefetching while
preserving its transforms and one-bag objective. Its schedule is 25 epochs total:
5 frozen, then 20 fine-tuning. Other methods' epoch counts are unchanged.
Validation never controls early stopping or checkpoint selection: the final model
is evaluated, and **no model weights are saved**.

The run name is `sample_efficiency_v1`. To replace the previous fixed-update run,
first stop its controller and any remaining jobs, then remove its results directory
on Palmetto before starting this sweep. This is required because saved settings
are immutable and the old results belong to a different training procedure.
From this sweep directory, run `rm -r -- ../results/sample_efficiency_v1`.
Outputs identify this design with `budget_mode=single_pass`. The legacy
`bags_per_reference_epoch` column now records the selected bags processed per epoch.
Smoke-test bag limits truncate the epoch and never repeat bags.

## Subsets

Patients are stratified by their bag-label signature (which also supports
patients with bags from multiple classes). Within each stratum, one deterministic
shuffle is used to take nested prefixes. All bags and images of a selected patient
are retained. The patient count in each stratum is rounded **up**, ensuring all
classes remain represented. Actual counts can slightly exceed the requested
percentage, especially in small datasets; inspect `subset_counts.csv` and plot
actual patient counts alongside percentages. There is no image sampling or bag
inflation in this experiment.

All eight methods use exactly the same saved subsets and validation patients.
The default is one subset sampling seed (`[123]`) for this first sweep. Additional
seeds can be added in a new run; they repeat subset selection, not the CV split.
Model initialization uses the split seed plus fold number, matched across sizes.
Ten CV folds are not ten independent repetitions.

## Run on Palmetto

From the repository, enter `Image Datasets/sample-efficiency-sweep`. Update the
data paths in the existing `../config/*_config.yaml` files if necessary. Confirm
`python` in `sweep.yaml` points to your PyTorch environment. The supplied profile
copies your current medical sweep account, partition, and A100 selection; edit
`profiles/palmetto/config.yaml` if your allocation changes.

```bash
module load anaconda3/2023.09-0
source activate snakemake-slurm
snakemake --profile profiles/palmetto --dry-run --printshellcmds
snakemake --profile profiles/palmetto --jobs 20
```

The controller uses `snakemake-slurm`; training and preparation use the explicit
`pytorch` Python executable. Default scope: **128 training jobs (ten folds sequentially per job)**, four
feature-extraction jobs, preparation, and summarization. Each training job runs all
ten folds, retaining completed folds if interrupted. Keep the
controller in a persistent session. To resume, repeat the same command; do not
unlock while another controller is still running.

For a short end-to-end test, use a separate run name and reduced budgets:

```bash
snakemake --profile profiles/palmetto --dry-run \
  --config run_name=sample_efficiency_smoke epochs_override=1 smoke_bags_per_epoch=2
snakemake --profile profiles/palmetto --jobs 2 \
  --config run_name=sample_efficiency_smoke epochs_override=1 smoke_bags_per_epoch=2
```

This retains the full grid and validation folds but trains for only two bags per
fold. It still needs feature extraction and must not be used for scientific
conclusions. Full runs use null overrides, as supplied. A run's resolved settings
and manifest are immutable: change `run_name` when changing experimental settings.
Code changes may trigger Snakemake reruns through normal input provenance.

## Outputs and checks

Outputs are in `../results/sample_efficiency_v1/`:

- `summary.csv`: 128 method/size/dataset summaries, mean validation AUROC and
  approximate 90% fold CIs, sensitivity, specificity, balanced accuracy, Brier,
  and separate pooled out-of-fold metrics. Use the mean AUROC for consistency
  with existing tables. Binary/multiclass Brier definitions match the comparator
  workflow; Brier is not a measure of calibration alone.
- `fold_metrics.csv`, `predictions.csv`: all held-out scores and predictions with
  patient/bag IDs, fold, method, percentage, and sampling seed for paired analyses.
- `manifest.json`, `subset_counts.csv`, `preparation_report.json`: exact bag
  membership, counts, and data provenance. Compare these patient assignments
  with archived runs before combining old and new results.
- `runs/<condition>/<method>/fold<N>/`: configuration, scores, predictions, and
  split manifest for each completed fold.
- `logs/<condition>/<method>/fold<N>.log`: Python output/errors. Slurm executor
  logs also appear under `.snakemake/slurm_logs/`.

Local design tests do not open medical images or train models:

```bash
python -m unittest discover -p 'test_*.py' -v
```

The existing comparator gradient tests can also be run in the PyTorch environment
from `../comparison-sweep`. Source/license notices remain in that directory.

Each training job covers one dataset, patient percentage, sampling seed, and method.
Together with four feature-extraction jobs, the default sweep submits 133 Slurm jobs, including one CPU preparation job.
Completed folds retain their own markers and checksums. Resubmitting an interrupted
job skips intact folds whose code, settings, and prepared data still match; an
unfinished fold restarts from its first epoch. Models are not saved.
The combined job log is `logs/<condition>/<method>/all_folds.log`; detailed fold
logs remain separate. The default time limit is 6 hours per job.

Preparation is submitted to Slurm as a CPU job (one CPU, no GPU), using the
configured memory and six-hour time limit. Only summarization runs locally
on the controller. Training and feature extraction continue to request GPUs.

All submitted jobs (preparation, feature extraction, and training) request 32 GB
of host memory via `mem_mb: 32000`. This does not set GPU memory.

## Instance and standard-assumption baselines

Run the additional baselines with `sweep_baselines.yaml`. The `instance` baseline
trains image predictions against inherited bag labels and evaluates each bag by
the arithmetic mean of its image probabilities. It does not average logits or
train on the averaged bag probability. `stan` is the standard-assumption baseline:
it trains against bag labels using the existing numerically stable noisy-OR
pooling. For multiclass X-ray, noisy-OR class scores are normalized across classes.

Both use a frozen ImageNet encoder, a randomly initialized trainable classifier,
last-epoch evaluation, and no saved model weights. Every selected image occurrence
is visited once per epoch; there is no repetition to pad out small training subsets.
Instance loss is averaged over the images in each training batch; larger bags
therefore supply more image loss terms. Validation losses and metrics are bag-level.
Dataset preprocessing, augmentation, optimizer, and schedules are inherited from
the saved source manifest. Training and validation losses for the instance baseline
represent different objectives and should not be interpreted as a generalization gap.

`source_run: sample_efficiency_v1` imports the exact saved subsets, validation folds,
and dataset configuration snapshots. Change this name if your completed run has a
different name. Missing or incompatible source files cause an error; subsets are
not silently regenerated. Outputs go to `results/sample_efficiency_baselines_v1`.

From this directory on Palmetto:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_baselines.yaml --dry-run
snakemake --profile ../palmetto-sweep/profiles/palmetto --configfile sweep_baselines.yaml --rerun-incomplete
```

This submits 32 GPU training jobs (4 datasets x 4 fractions x 2 baselines), each
running all ten folds, plus one CPU preparation job. Summarization runs locally.
Each training job requests 32 GB and six hours. Summary, fold metrics, predictions,
and subset counts retain the existing CSV formats; baseline method names are
`instance` and `stan`. Existing experiment results are not overwritten.
