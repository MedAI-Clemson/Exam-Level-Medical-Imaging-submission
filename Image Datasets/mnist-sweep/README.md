# MNIST bag composition experiment

This separate sweep reuses `../models.py` and `../pretraining.py`. The existing medical training code and palmetto-sweep configuration are unchanged.

The default experiment has **40 training jobs**: five pooling methods × bag sizes 20/100 × positive-instance densities 5/10/25/50 percent. Each uses one seed, 1,000 training bags, 200 validation bags and 1,000 test bags, with equal positive/negative counts. A positive bag contains exactly the specified fraction of digit 9; a negative bag contains no 9. Distractor digit classes are sampled uniformly from the other nine classes. Image sampling is with replacement within a source split, so images can recur across bags. These are synthetic bags, not independent patients.

Source images are split before constructing bags: a stratified 80/20 split of the official MNIST training images supplies training/validation; official MNIST test images supply testing. All methods share the same saved manifests. Each condition is trained and tested at the same density. This is not a test of generalization to unseen densities.

All methods use ImageNet ResNet-50, a trainable encoder, a random classifier, 16 attention heads where applicable, 224×224 RGB copies of grayscale images, ImageNet normalization, and no augmentation. Defaults: 20 epochs, Adam (learning rate 0.001, weight decay 0.001), StepLR (8 epochs, factor 0.1), and one bag per batch. Last epoch is evaluated. No model weights are saved. Change these settings in this sweep's YAML and use a new run name when changing the experiment.

## Palmetto commands

From the cloned repository:

```bash
cd "Image Datasets/mnist-sweep"
module load anaconda3/2023.09-0
source activate snakemake-slurm

# Review the 40 training jobs plus preparation and summary.
snakemake --profile ../palmetto-sweep/profiles/palmetto --dry-run

# Download MNIST, cache ImageNet weights, and build manifests on the controller.
snakemake --profile ../palmetto-sweep/profiles/palmetto prepare --cores 1

# Launch the full experiment, preferably in your persistent controller session.
snakemake --profile ../palmetto-sweep/profiles/palmetto
```

Edit `python` and `data_dir` in `sweep.yaml` if needed. Relative project/results paths are resolved from this directory. This reuses the existing profile's partition, account, GPU type and maximum concurrent jobs. The training interpreter is your `pytorch` environment; the controller environment is `snakemake-slurm`. Preparation needs internet access for MNIST and encoder weights, and the same home/cache must be visible on compute nodes. The controller only prepares data and summarizes; it does not train.

For a one-epoch smoke test of all 40 conditions, use a different run name. Repeat the same overrides when resuming that test:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --dry-run --config run_name=mnist_composition_smoke num_epochs=1
snakemake --profile ../palmetto-sweep/profiles/palmetto --config run_name=mnist_composition_smoke num_epochs=1
```

The effective configuration is saved even on a dry-run so command-line overrides reach every job. Do not run two controllers with different settings under the same run name.

## Outputs and interpretation

Outputs go to `../results/mnist_composition_v1/`:

- `design.json`: source splits and every bag's image indices, label and target count. Indices for train/validation address official training MNIST; test indices address official test MNIST.
- `runs/n20/p5/avg/` (and analogous paths): resolved configuration, epoch training/validation history, test predictions, and summary.
- `completed/n20/p5/avg.json`: success marker written only after predictions and metrics.
- `logs/`: preparation and per-job console output; SLURM logs remain in `.snakemake/slurm_logs`.
- `summary.csv`: 40 rows with test AUROC, Brier score and loss, generated automatically when all jobs finish.

Use density on the figure's x-axis, AUROC on the y-axis, separate panels for bag size and one curve per pooling method. No repetition-based CIs or significance tests are produced with this single-seed design. Source-image reuse also means ordinary bag bootstrap intervals should not be presented as independent-image uncertainty. Changing density changes both target count and distractor count; it is an experiment on bag composition under the standard MIL assumption.

Resume by rerunning the same command. Completed jobs are retained when inputs/settings are unchanged. Interrupted training restarts from epoch one because checkpoints are intentionally disabled. The initial resource/time estimates require calibration on Palmetto; the workflow is not intended for local training.
