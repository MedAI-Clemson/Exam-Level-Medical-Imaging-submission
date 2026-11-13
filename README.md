# Exam-Level Medical Imaging: MIL Code

Code for the paper **"Bridging Clinical Diagnosis and Deep Learning Workflows: An Analysis of Multiple Instance Learning for Exam-Level Classification"** (revised submission to IEEE Journal of Biomedical and Health Informatics).

The paper evaluates multiple-instance learning (MIL) methods for exam-level classification, where each imaging exam is a *bag* and its images (or video frames) are *instances*. This repository contains the training scripts, dataset classes, model definitions, configuration files and supporting code used in our experiments. File paths in the configs are left as used in our experiments, for completeness. See the paper for data availability.

## Repository layout

```
Image Datasets/        Image datasets: Achilles US, Prostate MRI, Shoulder X-ray, Lung CT
  achilles_train.py    Training script, Achilles tendon ultrasound
  mri_train.py         Training script, prostate MRI
  xray_train.py        Training script, shoulder X-ray
  ct_train.py          Training script, lung CT
  datasets.py          Bag-level (MIL*) and image-level dataset classes
  models.py            MILNet (pooling models) and StandardMILNet (noisy-OR baseline)
  transforms.py        Image transforms
  utils.py             Padding, parameter count, collate function, early-stopper helper
  config/              One MIL config and one instance-level pretraining config (*_pt_config.yaml) per dataset
Video_datasets/        Model, transform and utility code for the ultrasound video datasets (FAST, PDA)
```

## Methods in this code

`MILNet` in `Image Datasets/models.py` implements the instance embedding pooling methods. Select one with `--pooling-method` or `net_kwargs.pooling_method` in the config:

| Config value | Method in the paper |
| --- | --- |
| `avg` | Average pooling |
| `max` | Max pooling |
| `attn` | ABMIL (dot-product) |
| `tanh_attn` | ABMIL (MLP) |
| `transformer` | Transformer-based pooling |
| `stan` | Standard-assumption (noisy-OR) baseline, via `StandardMILNet` |

The instance baseline and instance-level encoder pretraining use the `*_pt_config.yaml` files (`mil: False`), which train an image classifier with the exam label applied to every image.

## Running an experiment

Dependencies include PyTorch, torchvision, timm, scikit-learn, torcheval, pandas, NumPy, PyYAML and matplotlib.

1. Set the data paths (`data_dir`, `metadata_csv`) and output paths (`save_dir`, `pretrained_model_dir`) in the dataset's config file.
2. (Optional) Run instance-level pretraining. Pretraining is run separately within each cross-validation fold, and each fold's checkpoint is saved as `model_fold{k}.ckpt`:

   ```bash
   cd "Image Datasets"
   python achilles_train.py --config config/achilles_pt_config.yaml
   ```

3. Train the MIL model, choosing a pooling method:

   ```bash
   python achilles_train.py --config config/achilles_config.yaml --pooling-method max
   ```

Encoder initialization and freezing are controlled in the config:

| Config key | Meaning |
| --- | --- |
| `encoder_kwargs.pretrained` | ImageNet initialization of the encoder (`timm` `resnet50d`) |
| `pretrain_enc` | Load the encoder from the instance-level pretraining checkpoint of the same fold |
| `pretrain_fc` | Load the instance-level classifier weights |
| `freeze_enc` | Freeze the encoder during MIL training |
| `freeze_fc` | Freeze the final classifier |

## Training settings in the provided configs

All image-dataset configs use a ResNet-50d encoder (`timm`), the Adam optimizer, a learning rate of 1e-3, a step learning-rate schedule (`StepLR`) and 10-fold stratified cross-validation (seeded for reproducibility). They train for a fixed number of epochs.

| Config | Batch size | Epochs | Weight decay | LR step size / gamma |
| --- | --- | --- | --- | --- |
| `achilles_config.yaml` (MIL) | 15 | 30 | 1e-3 | 8 / 0.1 |
| `achilles_pt_config.yaml` (instance) | 75 | 30 | 1e-3 | 50 / 0.1 |
| `mri_config.yaml` (MIL) | 3 | 10 | 5e-4 | 50 / 0.1 |
| `mri_pt_config.yaml` (instance) | 75 | 15 | 5e-3 | 50 / 0.1 |
| `xray_config.yaml` (MIL) | 25 | 50 | 1e-3 | 5 / 0.1 |
| `xray_pt_config.yaml` (instance) | 75 | 50 | 1e-3 | 50 / 0.1 |
| `ct_config.yaml` (MIL) | 1 | 10 | 5e-3 | 50 / 0.1 |
| `ct_pt_config.yaml` (instance) | 50 | 10 | 5e-3 | 50 / 0.1 |

## Data

Prostate MRI (ProstateX) and Lung CT (LIDC-IDRI) are public collections from The Cancer Imaging Archive. The Achilles tendon, shoulder X-ray, FAST and PDA datasets were collected with clinical collaborators; see the paper for details and availability. This repository does not include any data.

## Citation

If you use this code, please cite the paper (citation to be added after publication).
