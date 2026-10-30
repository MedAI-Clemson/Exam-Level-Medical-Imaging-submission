# Palmetto experiment sweep

Run commands from `Image Datasets/palmetto-sweep`. The controller environment is
`snakemake-slurm`; the `python` path in `sweep.yaml` selects the training environment.
Keep these environments separate. Existing Slurm account, partition, and GPU
settings remain in `profiles/palmetto/config.yaml`.

## Combined experiment

```yaml
run_name: combined_randaugment_sweep
encoder_states: [trainable, frozen]
num_heads: 16
num_epochs: null
```

Use `[trainable]` or `[frozen]` for just one state. This replaces the old
`freeze_encoder` setting; do not include both settings. Dataset YAML `freeze_enc`
values are overridden per MIL job. Instance pretraining always trains its encoder.

The supplied selection has four datasets, five pooling methods, three sources,
and two encoder states: **120 MIL jobs plus 4 instance-pretraining jobs**. Each job
runs all ten folds sequentially. The final `all` rule adds one orchestration job
to dry-run totals. The profile's `jobs` setting controls simultaneous submissions.

Each dataset's instance-pretraining job runs once. Both encoder states and all
pooling methods use those same fold checkpoints. ImageNet and RadImageNet jobs
can start independently; instance-initialized MIL waits for the matching dataset.
This new sweep creates its own shared pretraining checkpoints, using RandAugment;
it does not automatically reuse checkpoints from older sweeps.

All sources use `resnet50` for RadImageNet compatibility. MIL classifiers start
fresh. The base dataset config supplies paths, preprocessing, augmentation and
encoder dropout. Instance configs supply optimization settings and batch size.
`num_epochs: null` preserves each stage's configured epochs; `1` overrides both.
`model_selection: last` selects the last epoch; `best_loss` selects lowest
validation loss. `save_models: false` disables MIL checkpoints, but instance
checkpoints are always saved because subsequent jobs require them.

Frozen encoders run in evaluation mode, including disabled encoder dropout and
fixed BatchNorm statistics. This update does not change that behavior.

## Launch on Palmetto

Verify shared paths, RadImageNet checkpoint location and Slurm resources first.
With the controller environment activated, inspect the jobs without submitting:

```bash
snakemake --profile profiles/palmetto --dry-run --printshellcmds
```

Then submit:

```bash
snakemake --profile profiles/palmetto
```

For a short test, use a separate `run_name`, `num_epochs: 1`, fewer datasets or
sources if desired, and keep both encoder states. Do not reuse that run name for
the full experiment. Keep the controller in a persistent session or allocation
consistent with current cluster policy.

## Outputs and restarts

Paths below are relative to `results_dir/run_name/`:

```text
completed/trainable/xray/imagenet/attn.json
completed/frozen/xray/imagenet/attn.json
completed/xray/instance_pretrain.json
logs/trainable/xray/imagenet/attn.log
logs/frozen/xray/imagenet/attn.log
logs/xray/instance_pretrain.log
attempts/trainable/xray/imagenet_attn/run_.../artifacts/
attempts/frozen/xray/imagenet_attn/run_.../artifacts/
attempts/shared/xray/instance_pretrain/run_.../artifacts/
```

Each completion JSON points to the successful attempt, and records the encoder
state. `experiment.yaml` is the generated trainer input; `artifacts/config.yaml`
records resolved settings. Keep attempt directories: they hold the predictions,
metrics and shared instance checkpoints referenced by completion markers.

Reissuing the command skips unchanged completed jobs. Changes to tracked code or
configuration can trigger reruns; retries start fresh attempts, not resumed epochs.
Use a new run name after changing data, augmentation or experimental settings.
Old sweep directories are not moved or deleted by this update.

To request one job, give its absolute marker path instead of the default target:

```bash
snakemake --profile profiles/palmetto /your/results/combined_randaugment_sweep/completed/frozen/xray/instance/attn.json
```

## Summaries

Run with the training environment's Python (requires pandas, NumPy, SciPy, PyYAML):

```bash
/home/bformby/.conda/envs/pytorch/bin/python summarize_results.py \
  '../results/combined_randaugment_sweep'
```

The command reads completed MIL markers recursively, verifies resolved encoder
state and complete folds, and writes `auroc_90ci.csv`, `brier_90ci.csv`, and
`metrics_per_fold.csv` inside that sweep folder. Re-running replaces these derived
summaries. Both encoder states are included as separate rows, never averaged
together. It also supports old single-state sweeps by reading `freeze_enc` from
their resolved configs. Unfinished attempts without completion markers are ignored;
inconsistent or missing files referenced by a completed marker produce an error.

Summaries use the mean across folds and descriptive 90% t intervals. These do not
correct for overlapping cross-validation training sets or multiple comparisons,
and are not significance tests. Binary Brier uses squared positive-probability
error; multiclass Brier sums squared class errors (range 0–2). AUROC uses the
trainer's stored fold metric, including macro one-versus-rest for X-ray.

Training, GPU submission and a full Snakemake dry-run must be checked on Palmetto.
