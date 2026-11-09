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
- ProMIL: reuse its existing ImageNet ResNet-18 medical profile, including 25
  frozen reference epochs followed by 100 fine-tuning reference epochs. This
  comparator is **not frozen throughout**. X-ray uses the existing OVR extension.
  No method loads an instance-pretrained encoder or classifier.

One reference epoch processes the **full training fold's bag count** at every
fraction, using shuffled repeated passes over smaller subsets. For the five
pooling methods, batch sizes and the short final batch are also fixed, so the
number of optimizer updates is identical across fractions within a fold/method.
CLAM/ProMIL take one bag per update; DTFD performs its existing two-tier updates
per bag. Learning-rate milestones and ProMIL unfreezing occur at the same update
counts across fractions. The budgets differ between methods, as in the existing
comparison. Validation never controls early stopping or checkpoint selection:
the final model is evaluated, and **no model weights are saved**.

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
Together with four feature-extraction jobs, the default sweep submits 132 Slurm jobs.
Completed folds retain their own markers and checksums. Resubmitting an interrupted
job skips intact folds whose code, settings, and prepared data still match; an
unfinished fold restarts from its first epoch. Models are not saved.
The combined job log is `logs/<condition>/<method>/all_folds.log`; detailed fold
logs remain separate. The default time limit is 6 hours per job.
