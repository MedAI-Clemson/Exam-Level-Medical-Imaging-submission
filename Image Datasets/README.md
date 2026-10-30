# Image classification workflow

One trainer supports Achilles ultrasound, shoulder X-ray, prostate MRI, and lung
CT. `mil: true` trains on bags; `mil: false` trains an instance classifier on the
same patient folds. The four original script names are compatibility launchers.

## Run on the training machine

Use Python 3.10+ and install matching PyTorch/torchvision builds for that machine.
Then, from this directory:

```sh
pip install -r requirements.txt
python train.py --config config/ct_config.yaml --check-config
python train.py --config config/ct_config.yaml
python xray_train.py --config config/xray_pt_config.yaml
python -m unittest discover -s tests -v
```

The code was refactored without access to the source datasets or a training
runtime. The supplied tests use synthetic data and do not download weights.
Validate the metadata and preprocessing on the intended machine before experiments.

All example paths resolve relative to this directory, regardless of the shell's
working directory. To use another base, set `path_root` in YAML; it resolves
relative to the YAML file. Absolute paths also work. Use `--save-dir` for a new
output location; an existing nonempty output directory is never overwritten.
`--device`, `--pooling-method`, and `--num-heads` override YAML settings.
`--pretraining` selects the encoder's weight source. `--freeze-encoder` and
`--no-freeze-encoder` override `freeze_enc` without editing the YAML.

## How the code fits together

| File | Responsibility |
| --- | --- |
| `configuration.py` | Read YAML, check settings, resolve paths |
| `datasets.py` | Adapt source CSVs, create a manifest, lazily load samples |
| `imaging.py` | Order DICOM slices and decode/normalize pixels |
| `transforms.py` | Resize, training RandAugment, input normalization |
| `models.py` | Shared encoder and instance/MIL classifiers |
| `pretraining.py` | Weight-source settings and RadImageNet checkpoint loading |
| `training_options.py` | Final epoch selection and model-saving settings |
| `train.py` | Cross-validation, training, checkpointing, prediction export |
| `utils.py` | Patient splits, metrics, seeding, pretrained-weight checks |

Classifier code consistently uses `clf`: the module is `model.clf`, its input
width is `clf_features`, and its settings are `instance_load_clf` and `freeze_clf`.
"Head" refers only to an attention head (`num_heads` / `--num-heads`). The
transformer's learned bag-summary token is named `bag_token`.

Every sample contains `[images, channels, height, width]` plus its patient ID,
bag ID, and label. Collation concatenates images and preserves bag lengths. The
model always returns logits shaped `[samples, outputs]`, including single-sample
batches. Binary tasks use BCEWithLogitsLoss; X-ray uses cross-entropy. Probabilities
are calculated only for metrics and exports, except internally for noisy-OR.

## Dataset conventions

| Task | Patient grouping | Bag | Labels |
| --- | --- | --- | --- |
| Achilles | `Patient #` | `Folder Name` (visit) | Healthy = 0, unhealthy = 1 |
| X-ray | `patient_id` | Patient | 1 part = 0; 2 parts = 1; 3/4 parts = 2 |
| MRI | `ProxID` | Patient across selected series | `ClinSig` (0/1) |
| CT | `PatientID` | `SeriesUID` | Diagnosis 0 = 0; 1/2 = 1 |

Achilles retains the original `Drop`, duplicate-image, and bilateral-folder
exclusions. X-ray joins images to targets explicitly on `patient_id` and checks
overlapping labels. MRI/X-ray view filtering happens before splitting; bags with
no eligible images are reported and excluded. Conflicting labels within a bag,
missing IDs, shared image paths across bags, and empty series raise errors.
Resolve ambiguous bag labels in the metadata; the loader does not select the
first label or silently average conflicting annotations.

Folds are stratified at bag level and grouped by patient. All visits from the
same patient remain together. Grouping can make exact class balance impossible;
the workflow rejects folds missing a class and recommends fewer folds. It requires
at least `folds` distinct patients per class. Instance pretraining expands bags
only after the split. Its metrics are **per image**; MIL metrics are **per bag**.

## Preprocessing choices

DICOM decoding preserves float32 intensities and applies the modality LUT or
rescale slope/intercept. CT uses the explicit `preprocessing.ct_window` bounds
in HU, clips, and scales to [0, 1]. The example `[-1000, 400]` is a configurable
experiment preset, not an inferred optimum or a reproduction of the old uint8
conversion. MRI uses per-slice percentile clipping, default `[1, 99]`, then scales
to [0, 1]. MONOCHROME1 images are inverted after scaling. Constant MRI slices are
handled without division by zero. PNG/JPEG intensities scale by their integer
dtype range. Both training and validation then normalize using mean/std 0.5.

Training uses fresh RandAugment on every image access, for all datasets and both
instance and MIL training. Defaults match the original `RandAugment()`: two
operations, magnitude 9, nearest-neighbor interpolation and default zero fill.
After resizing, scaled training pixels are temporarily quantized to uint8 for
RandAugment, then converted back to [0, 1] floats before normalization. DICOM
windowing/percentile scaling still happens before this conversion. Validation
has no random augmentation or uint8 conversion.

```yaml
res: 224
transform_kwargs:
    num_ops: 2 # Set 0 to disable augmentation
    magnitude: 9 # Integer from 0 to 30
    mean: 0.5
    std: 0.5
```

Replace legacy `transform_kwargs.rotation` with the options above. Attention
defaults to 16 heads in resolved configs and the sweep; `net_kwargs.num_heads`
or `--num-heads` can explicitly override it. Use a new sweep run name when
comparing this augmentation with existing results.

Slices are ordered by position projected onto the orientation-derived normal,
with InstanceNumber as the fallback when geometry is absent. Duplicate positions,
inconsistent orientations, and multiple series in one directory are rejected.
Only single-frame, grayscale DICOM files with a `.dcm` extension are supported.
Compressed transfer syntaxes may need additional pydicom decoder packages.

The optional MRI `use_k` setting requires both an explicit index base and an
ordering convention that matches the annotation source:

```yaml
use_k: true
k_index_base: 0
slice_order: position # or instance
```

The third `ijk` coordinate indexes that ordered list. It is not assumed to be a
filename or an InstanceNumber value. The selected paths are k-1, k, k+1 within
bounds, including slice zero. If the annotation volume uses another orientation
or reverse ordering, map its coordinates before enabling this option. This
option uses finding annotations to select inputs; report that use when evaluating.

## Pooling and checkpoints

Supported pooling methods: `attn`, `tanh_attn`, `avg`, `max`, `transformer`, `stan`.
Padding is excluded from pooling. The transformer has a 64-dimensional projection.
Attention head counts must divide the feature width; transformer head counts must
divide 64. Standard MIL (`stan`) uses noisy-OR in log space, with class-wise
softmax for multiclass inputs and normalized noisy-OR class scores at bag level.
Noisy-OR can still saturate for large bags; that is a modeling property.

Each fold selects its final model using `model_selection: last` (the default) or
`model_selection: best_loss` (lowest validation loss; ties keep the earlier epoch).
Final metrics and predictions come from the selected epoch. Frozen encoders keep
BatchNorm statistics and dropout frozen as well as parameters.

`save_models` defaults to `true`. When enabled, `model_foldN.ckpt` contains the
selected model's named encoder/classifier weights, configuration, epoch, metrics,
and train/validation patient IDs. Last-epoch mode saves only at the end of the fold;
best-loss mode replaces the same checkpoint on each improvement.

The supplied MIL configs currently set `save_models: false` to reduce disk usage.
Metrics, predictions, split manifests, and configuration files are still written.
No model checkpoint is created, even temporarily, and no extra weight snapshot is
kept in RAM: the trainer retains the selected epoch's validation outputs instead.
This does not reduce the GPU memory needed for training. To save models again:

```sh
python train.py --config config/ct_config.yaml --model-selection best_loss --save-models
python train.py --config config/ct_config.yaml --model-selection last --no-save-models
```

The `*_pt_config.yaml` files retain `save_models: true`, because their checkpoints
are inputs to instance-initialized MIL runs. Standalone instance training can use
`--no-save-models`, but those results cannot initialize a later model.

## Three encoder initialization options

`pretraining` selects where the encoder weights come from; `mil` selects whether
training predicts one label per bag or one label per image. These are separate
settings. Choose exactly one initialization source:

| `pretraining` | Encoder initialization | Additional settings |
| --- | --- | --- |
| `imagenet` | timm's pretrained ImageNet weights for the configured architecture | No local checkpoint required |
| `radimagenet` | Local official PyTorch RadImageNet ResNet-50 weights | `radimagenet_checkpoint`; `model_name: resnet50` |
| `instance` | The matching fold's classifier trained on images with inherited bag labels | `pretrained_model_dir`; `mil: true` |

### ImageNet

```yaml
pretraining: imagenet
freeze_enc: False
```

The example ImageNet configs retain `resnet50d`. The encoder's old `pretrained`
boolean is no longer needed: the source selector controls it. Only ImageNet mode
requests pretrained weights from timm. RadImageNet and instance modes construct
the architecture without downloading ImageNet weights first.

### RadImageNet

```yaml
pretraining: radimagenet
radimagenet_checkpoint: "./weights/resnet50_torch.pt"
encoder_kwargs:
    model_name: resnet50
    in_chans: 1
    num_classes: 1 # Use 3 for X-ray
    drop_rate: 0.0
freeze_enc: False
```

Download the PyTorch weights linked by the [official RadImageNet repository](https://github.com/BMEII-AI/RadImageNet)
on the training machine and point the config at the extracted ResNet-50 checkpoint.
The workflow does not automatically download or convert RadImageNet files.
`config/ct_radimagenet_config.yaml` is a complete example.

This implementation supports the official ResNet-50 `backbone.*` parameter layout,
or a standard named ResNet-50 state dictionary, optionally wrapped in `state_dict`
or `model_state_dict`. It loads strictly and rejects incompatible layers. Other
RadImageNet architectures and TensorFlow/Keras `.h5` files are not supported.
Standard `resnet50` is required; `resnet50d` has a different convolutional stem.
For a controlled comparison, use `resnet50` in the ImageNet and instance-pretraining
configs too, keeping the architecture fixed across initialization sources.

The RGB first-layer kernels are summed into one grayscale kernel. This is
equivalent to repeating the normalized grayscale image across three channels.
Mean/std 0.5 map inputs to [-1, 1], consistent with the scaling in the
[official PyTorch example](https://github.com/BMEII-AI/RadImageNet/blob/main/pytorch_example.ipynb).
The existing CT windowing and MRI intensity normalization still happen first.

### Instance pretraining with bag labels

This is a two-stage experiment. First, train an instance classifier:

```yaml
mil: False
pretraining: imagenet # Or radimagenet with its architecture and checkpoint settings
save_dir: "./ct/pretrain"
freeze_enc: False
```

Each image receives its bag's label; no separate image annotations are required.
For example, every slice in a positively labeled CT bag has target 1 during this
stage. Patient splitting happens before creating instance samples, preventing
images from a validation patient from entering the training partition.
The supplied `*_pt_config.yaml` files implement this stage.

Then initialize the MIL model with the resulting fold checkpoints:

```yaml
mil: True
pretraining: instance
pretrained_model_dir: "./ct/pretrain"
instance_load_clf: False # Set true to transfer the instance classifier as well
freeze_enc: False
```

`model_fold1.ckpt` initializes MIL fold 1, and so on. The encoder is always loaded;
the classifier is optional. Matching task, architecture, output count, and
patient partitions are checked before transfer. A transformer classifier cannot load
the instance classifier because their feature widths differ. Keep the data,
view filters, preprocessing, seed, fold count, and architecture consistent across
the two stages. The instance encoder may itself have started from ImageNet or
RadImageNet; the saved configuration records that history.

For Achilles, run `achilles_pt_config.yaml` followed by `achilles_config.yaml`;
the latter already selects instance weights and classifier transfer. Other MIL configs
default to ImageNet and can select either alternative. Older config flags
`pretrain_enc` and `pretrain_fc` are translated to the explicit instance settings.
The previous `instance_load_head` and `freeze_fc` options also remain readable,
but resolved configs use `instance_load_clf` and `freeze_clf`. Conflicting old and
new values are rejected. Old `head.*` classifier weights and the old `cls` token
name are mapped when loading checkpoints; new checkpoints save `clf.*` and
`bag_token`. The old `--instance-load-head` / `--no-instance-load-head` CLI spellings
remain accepted as hidden compatibility aliases for the new `clf` spellings.
Version-2 checkpoints from this refactored workflow remain compatible. Checkpoints
from the original pre-refactor scripts lack compatible names and split provenance
and must be regenerated.

### Encoder freezing

Set `freeze_enc: True` with any of the three sources to train only the unfrozen
pooling/classifier parameters. Encoder weights, BatchNorm statistics, and dropout remain
fixed even when the trainer calls `model.train()`. Set it to `False` for encoder
fine-tuning. The separate `freeze_clf` setting continues to control the classifier.

```sh
python train.py --config config/ct_config.yaml --freeze-encoder --save-dir ./test/ct_frozen
python train.py --config config/ct_radimagenet_config.yaml --freeze-encoder
python train.py --config config/ct_config.yaml --pretraining instance --pretrained-model-dir ./ct/pretrain
```

`--instance-load-clf` / `--no-instance-load-clf` control optional classifier transfer.
`--radimagenet-checkpoint` overrides the RadImageNet path. Switching the source to
ImageNet or RadImageNet on the command line disables instance classifier transfer unless
explicitly requested (an incompatible request is rejected).

The standard-library configuration/mapping tests can run independently:

```sh
python -m unittest discover -s tests -p test_pretraining_config.py -v
```

The full test suite also checks encoder freezing, successful instance transfer,
bag-label inheritance, and strict loading from a synthetic official-style
ResNet-50 checkpoint. It requires the training dependencies, but no downloads.

## Outputs and evaluation

Each run writes `config.yaml`, the full `manifest.csv`, train/validation manifests
per fold, epoch metrics, predictions with IDs and probabilities, and `summary.csv`.
Model checkpoints are written only when `save_models: true`. The summary records
`selected_epoch` and `model_selection`; `selected_epoch` replaces the old
`best_epoch` column because the selected epoch need not minimize loss.
Binary `probability_1` means the positive class. Multiclass
probabilities use `probability_0` through `probability_2`. Undefined AUROC is recorded
as NaN (blank in CSV), never zero. Full training cannot proceed with class-incomplete
folds; the metric helper still handles such subsets when used independently.

Selecting a checkpoint and reporting metrics on the same validation fold is
validation performance, not an untouched test estimate. Use a separate test set
or nested validation if unbiased model-selection evaluation is needed. This
rewrite changes preprocessing, augmentation, patient grouping, and checkpoint
selection, so historical experiment results must be rerun rather than compared
as though only the code layout changed.

When launched through the Snakemake wrapper, `experiment.yaml` is the generated
input to train.py: a dataset config with that experiment's sweep overrides.
The trainer's `artifacts/config.yaml` is the validated, resolved version actually
used, including defaults, canonical option names, and resolved paths. Prefer
`artifacts/config.yaml` when inspecting or reproducing a completed run; set a new
save directory before rerunning it. These small YAML files contain no model weights.

Implementation references: [pydicom modality LUT](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html),
[timm feature extraction](https://huggingface.co/docs/timm/feature_extraction),
[patient-grouped stratification](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
