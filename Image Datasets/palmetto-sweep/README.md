# Palmetto experiment sweep

This workflow calls the existing Image Datasets trainer. It lives at the repository
root beside Image Datasets. Run Snakemake commands from this folder on Palmetto.

## Current smoke-test preset

The supplied sweep.yaml selects ImageNet only, all four datasets, all six pooling
methods, one epoch, and two folds. This produces 24 GPU jobs and 48 one-epoch fits.
The profile submits at most two jobs at once. No instance-pretraining job runs,
no RadImageNet checkpoint is required, and no model checkpoints are saved.
Metrics, predictions, and configuration files are still saved. Each epoch uses
the full dataset; this is an execution test, not a performance comparison.

`num_epochs: 1` overrides both MIL and instance-stage epochs when those stages are
selected. You do not need to edit the dataset configs. Set it to null to restore
each config's epoch count. Select initialization sources with `sources` in
sweep.yaml. For the full experiment below, use `[imagenet, radimagenet, instance]`,
set folds to 10, restore the desired epoch counts, and choose a new run_name.

## Full experiment design

- Four datasets: Achilles, CT, MRI, X-ray.
- Six pooling methods: attn, tanh_attn, avg, max, transformer, stan.
- Three MIL initialization sources: ImageNet, RadImageNet, instance pretraining.
- 72 MIL jobs plus four instance-pretraining jobs = 76 GPU jobs before retries.
- Every job runs all folds sequentially. With ten folds, this is 760 model fits.
- One GPU per job; the profile's jobs setting controls concurrent submissions.

ImageNet/RadImageNet MIL jobs can start immediately. Each dataset's six instance
MIL jobs wait for that dataset's instance-pretraining job. Pretraining is shared
across pooling methods and starts from ImageNet by default.

All sources use resnet50 because the supported RadImageNet checkpoint cannot
initialize resnet50d. All MIL classifiers start fresh (`instance_load_clf: false`),
including Achilles, so the transformer is compatible and transfer settings are
consistent. The instance stage always fine-tunes its encoder; `freeze_encoder`
controls the later MIL stage for all three sources.

`model_selection: last` selects the final epoch by default. Set it to `best_loss`
to select the epoch with the lowest validation loss (earliest epoch on a tie).
The setting applies to both stages. Summary files report `selected_epoch` and
`model_selection`; predictions also identify their selected epoch.

The trainer and wrapper default `save_models` to true when omitted. The supplied
sweep explicitly sets it to false for your current MIL experiments. All metrics,
predictions, and configs are retained; MIL jobs create no checkpoint files.
The four instance-pretraining jobs still save their fold checkpoints because
downstream MIL jobs require those weights. With ten folds, this leaves 40 required
instance checkpoints instead of 760 total model files. No existing files are deleted.
Disabling saving reduces disk storage, not the GPU memory required for training.

The MIL config supplies each dataset's paths, filters, preprocessing, and encoder
settings for both stages. The PT config supplies only instance-stage epochs,
optimizer, scheduler, and batch size. Shared seed and fold count come from
sweep.yaml. This prevents accidental patient-split differences between stages.
Normalization is inherited unchanged from the current workflow.

## Files

- `Snakefile`: experiment combinations, dependencies, and GPU rules.
- `sweep.yaml`: paths, experiment settings, and resource estimates.
- `profiles/palmetto/config.yaml`: Slurm executor and submission limits.
- `run_experiment.py`: generates one resolved config and invokes train.py.

## Configure on Palmetto

1. Replace all `/REPLACE/...` paths in sweep.yaml with absolute shared Linux paths.
   The Python executable must belong to an environment with the training
   dependencies, compatible CUDA PyTorch, and PyYAML. Activate that environment
   and load any site modules it requires before launching Snakemake.
2. Update the four `config/*_config.yaml` dataset paths in your Palmetto checkout.
   Original relative paths work if you preserve the expected directory layout;
   absolute paths are simpler on the cluster. The PT configs remain present for
   their instance optimization settings.
3. When enabling RadImageNet, put its official PyTorch ResNet-50 checkpoint at the
   configured path. The current ImageNet-only test does not use that path.
   ImageNet weights must be downloadable from compute nodes or already
   cached in the environment's shared timm/Hugging Face cache.
4. Adjust CPU, memory, wall time, and batch sizes after a representative pilot.
   Memory values are host RAM, not GPU VRAM. `runtime_minutes: 1440` is an estimate
   for the entire cross-validation job, not each fold, and is not a statement of
   your account's wall-time limit. Match requests to your allocation's limits.
5. The profile uses work1. Set your actual `slurm_account` if automatic account
   discovery is unsuitable. Optionally set `gpu_model` for a consistent GPU type
   or use a suitable owner partition. No account or GPU type is assumed here.

The Slurm executor supports `gpu`, `mem_mb`, and `runtime` resource mappings.
See the [official executor documentation](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html).
Palmetto documents `--gpus` requests and partition selection in its
[job submission guide](https://docs.rcd.clemson.edu/palmetto/job_management/submit/).
Its general batch partition is work1; consult the
[scheduler guide](https://docs.rcd.clemson.edu/palmetto/job_management/sched/)
for preemption and owner partitions.

## Launch

Install the controller packages in the activated shared environment on Palmetto:

```bash
python -m pip install 'snakemake>=9,<10' snakemake-executor-plugin-slurm
```

Check the graph first; this does not run training or submit GPU jobs:

```bash
cd /your/shared/path/palmetto-sweep
snakemake --executor local --cores 1 --dry-run --printshellcmds
```

Then submit through the profile:

```bash
snakemake --profile profiles/palmetto
```

Snakemake calls sbatch itself. Do not put sbatch inside the training rules.
Keep the controller alive for the whole sweep in a persistent session at a
location permitted by Palmetto's current usage policy. The controller submits
and monitors jobs; training runs on allocated compute nodes. If a controller
batch job is required, its time limit must cover queue delays and the full sweep,
not just one training run. Record installed package versions with your results.

For an initial single MIL job, request its absolute completion marker:

```bash
snakemake --profile profiles/palmetto \
  /your/results/smoke_imagenet_1epoch_v1/completed/ct/imagenet/avg.json
```

Replace the target's prefix with your actual results_dir and run_name. To test
the instance dependency later, add instance to sources and use
`completed/ct/instance/avg.json`.

## Results and restarts

Each successful job writes a JSON manifest under
`results_dir/run_name/completed/`. It points to the full training outputs,
including predictions, summary.csv, and checkpoints when enabled, in a unique directory
under `attempts/`. Training logs append under `logs/`. Keep the attempt directories:
the JSON files refer to them and instance transfer reads their checkpoints.

`experiment.yaml` is the generated input configuration passed to train.py.
`artifacts/config.yaml` is the trainer's validated, resolved configuration after
defaults and path resolution. Use the latter to inspect the exact settings used.
Both are small text files and neither contains weights. The Slurm profile's
`profiles/palmetto/config.yaml` is separate and controls job submission.

Reissuing the same Snakemake command skips completed, unchanged experiments.
Failed attempts remain available for diagnosis. The profile disables automatic
Slurm requeue and uses zero automatic retries; after fixing the cause, rerun
Snakemake to start the failed experiment from the beginning. The model does not
resume from its last epoch. Snakemake's incomplete-job handling is already enabled.

The rules track source files, config files, and local RadImageNet weights as
inputs. They do not enumerate every image or fingerprint remote ImageNet weights.
Use a new run_name after changing data or experimental settings, and never remove
an attempt directory referenced by a completion marker. If outputs are removed,
remove the corresponding completion marker and rerun the affected experiment.

To compare frozen versus fine-tuned encoders, run a second sweep with
`freeze_encoder: true` and a new run_name. This template generates a separate set
of four instance pretraining runs for that sweep.

If all folds do not fit in one allocation, the next code change should add a
fold selector and separate fold output paths to train.py. A Snakefile alone
cannot parallelize its current internal fold loop correctly.

This template has not been submitted to Palmetto or parsed by Snakemake here.
Only Python syntax and isolated configuration/wrapper checks were performed.
