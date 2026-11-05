# MNIST bag composition sweep

This workflow reuses `../models.py` and `../pretraining.py`, and the existing Palmetto profile. Training remains ImageNet ResNet-50 with a frozen encoder, 10 epochs, last-epoch evaluation, and no saved models, as configured in `sweep.yaml`.

## Requested conditions

| Bag size | Positive-bag digit 9 densities | Training jobs |
|---|---|---|
| 10 | 10%, 20%, 50% | 15 |
| 20 | 5%, 10%, 25%, 50% | 20 |
| 100 | 1%, 5%, 10%, 25%, 50% | 25 |

There are 60 runs across five pooling methods. The previous v2 sweep had 30 runs (sizes 20/100 at 5/10/50%). Retaining those leaves **30 new GPU jobs**. Each condition uses 1,000 training, 200 validation, and 1,000 test bags. Each split is balanced. The target digit is 9, negative bags contain none, and distractors are sampled uniformly over the other digit classes. Source-image splits and sampling seeds are unchanged from the previous implementation.

## Extend the existing v2 results on Palmetto

Push these changes, pull them on Palmetto, then run the following from the repository root. Stop any existing controller using this run folder before migrating.

```bash
cd "Image Datasets/mnist-sweep"
module load anaconda3/2023.09-0
source activate snakemake-slurm

# Run once, before the first dry-run of the expanded workflow.
/home/bformby/.conda/envs/pytorch/bin/python run_experiment.py migrate --config sweep.yaml

snakemake --profile ../palmetto-sweep/profiles/palmetto --dry-run
snakemake --profile ../palmetto-sweep/profiles/palmetto
```

Migration should report **30 retained runs** if all old runs are present and their saved settings match. The dry-run should then show **30 train_mnist jobs**, six local preparation jobs, and summary/all jobs. Do not proceed if completed runs unexpectedly appear as training jobs; investigate the migration message. The profile's controller `local-cores: 1` serializes preparation and shared-cache downloads.

Migration reads the original `design.json`, completion markers, resolved per-run settings and predictions. It checks model settings, frozen/trainable labels, epochs, design hashes, and prediction bag IDs/labels/counts. It writes checksummed receipts under `retained/`. It does not rewrite old manifests, scores, prediction files, or completion markers, and it does not retrain. Repeating migration is safe; it will not reapprove a retained run whose code, settings, or files have changed. A metadata mismatch stops migration rather than guessing which experiment was executed.

Retained receipts are explicit external inputs to the new summary, not outputs of the training rule. This avoids triggering legacy training because its old Snakemake metadata references the previous shared configuration. No global `--rerun-triggers mtime`, touching of completion files, or blanket suppression of dependency checks is needed. The stored code hashes establish the migration-time implementation baseline; they cannot prove the exact historical code used on Palmetto. Only migrate after confirming that the old runs used the model/training implementation you intended.

## New sweeps and later additions

For a fresh experiment, select a new `run_name` and run Snakemake normally; no migration is needed. For a one-epoch smoke test use a fresh name:

```bash
snakemake --profile ../palmetto-sweep/profiles/palmetto --config run_name=mnist_expansion_smoke num_epochs=1
```

To extend this grid later, add densities under the appropriate key in `densities_by_bag_size`. Remove the old `bag_sizes` and `densities` keys; do not mix the two formats. Every percentage must correspond to a whole image count. Each condition has an independent stable configuration and manifest, so adding a condition alone does not alter existing training inputs. Adding a pooling method from the supported five also leaves the existing condition configurations unchanged.

Changing training settings or model code is a different experiment: use a new run name. Retained results reject such changes; ordinary new-format jobs continue to use Snakemake dependency tracking. Resource/execution settings are not grounds for rejecting a retained result, though changing fields in a new-format condition configuration can trigger its rerun. Run from this directory so relative paths resolve consistently.

## Output layout

The run folder remains `../results/mnist_composition_v2/`:

- `design.json`, `sweep.resolved.json`, and old run files: preserved legacy records.
- `retained/n20/p5/avg.json` and analogous paths: verified legacy results with checksums.
- `sweep.expanded.json`: full current grid, used only for summary generation.
- `configs/n100/p1.json`: stable configuration for one condition.
- `designs/n100/p1.json`: newly generated bags for that condition, shared across methods.
- `runs/n100/p1/avg/`: resolved settings, epoch history, test predictions and summary.
- `completed/n100/p1/avg.json`: completion marker for a new training run.
- `logs/`: preparation/training console output. SLURM logs remain in `.snakemake/slurm_logs`.
- `summary.csv`: automatically rebuilt with retained and new results, 60 rows after completion.

Condition manifests have different hashes from the original combined manifest. That is expected: hashes identify different files, not different source split policies. Each row retains the hash of the manifest actually used for its training run.

Completed new-format jobs resume normally when inputs are unchanged. Interrupted training restarts at epoch one because checkpoints are disabled. No training is intended on the local workstation. CPU-only design/reuse checks can be run with `python -m unittest discover -p 'test_*.py'`.

Report these as single-seed results without repetition-based confidence intervals. At fixed density, larger bags contain more target and distractor images; this is a composition experiment, not an isolated causal test of density alone.
