# Video datasets (FAST, PDA)

Model, transform and utility code for the ultrasound video datasets, plus RadImageNet-initialised training for FAST and PDA.

```
models.py              Pooling models, PDA interface (frames grouped by study)
transforms.py          Video transforms
utils.py               Helpers (parameter count, early stopper, padding)
radimagenet.py         Loads RadImageNet ResNet50 weights into a timm `resnet50` encoder
PDA/                   PDA training (uses ../models.py)
  dataset.py             PdaVideos dataset and study-level collate function
  train_cv_radimagenet.py
  config_radimagenet.yaml
  launch_radimagenet.sh  SLURM launcher
FAST/                  FAST training
  dataset.py             VideoDataset and collate function (adds per-study indices)
  models.py              Pooling models, FAST interface (study_indices)
  train_cv_radimagenet.py
  config_radimagenet.yaml
  launch_radimagenet.sh  SLURM launcher
```

## RadImageNet experiments

Two training strategies from Table I of the paper, with the encoder initialised from RadImageNet instead of ImageNet and a randomly initialised classifier:

| Strategy | Encoder init | Encoder frozen | Flag |
| --- | --- | --- | --- |
| 1 | RadImageNet | No | (default) |
| 3 | RadImageNet | Yes | `--freeze-encoder` |

Each is run for the five pooling methods (`avg`, `max`, `attn`, `tanh_attn`, `transformer`), with 6-fold cross-validation, 12 epochs.

RadImageNet weights are not included. Download `ResNet50.pt` from <https://huggingface.co/Lab-Rasool/RadImageNet> and pass its path with `--radimagenet-weights` (or set `radimagenet_weights` in the config). It is a plain `resnet50` (not `resnet50d`), so the configs use `model_name: resnet50` with `pretrained: False`. `radimagenet.py` remaps the layer names into timm's `resnet50`.

The best weights of each fold are kept in memory only; the scripts do not write checkpoint files.

## Running

```bash
cd FAST   # or PDA
python train_cv_radimagenet.py config_radimagenet.yaml \
    --artifact-folder out/ --plot-dir out/ \
    --frame-csv /path/to/catalog.csv \
    --radimagenet-weights /path/to/ResNet50.pt \
    --pooling-method attn --epochs 12 [--freeze-encoder]
```

`launch_radimagenet.sh` submits all 10 runs (5 pooling methods x strategies 1 and 3) on SLURM after you set the three paths at its top.

The catalogs are not included. FAST expects one row per frame with `NewPatientID`, `VideoName`, `VideoPath` and `Quality`; PDA expects `patient_id`, `study`, `view`, `mode`, `patient_type` and `png_path` (see the dataset classes for details).
