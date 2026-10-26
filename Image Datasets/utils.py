"""Reproducibility, patient-isolated folds, metrics, and checkpoint selection."""

import random
import warnings

import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold

from pretraining import normalize_model_state_dict


def seed_everything(seed):
    # Seed all random sources used for sampling, augmentation, and model initialization.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Favor repeatable cuDNN behavior over automatic algorithm benchmarking.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    # DataLoader assigns each worker a torch seed; align its NumPy and Python RNGs.
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_folds(manifest, n_splits, seed):
    # Split bags before expanding to instances, so pretraining and MIL use the same patients.
    bags = manifest[["bag_id", "patient_id", "label"]].drop_duplicates().sort_values("bag_id")

    counts = bags.groupby("label").patient_id.nunique()
    if (counts < n_splits).any():
        raise ValueError(f"Need at least {n_splits} distinct patients per class; got {counts.to_dict()}")

    # Stratify using bag labels, but keep every bag from a patient in the same partition.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_index, val_index in splitter.split(bags, bags.label, bags.patient_id):
        train_patients = set(bags.iloc[train_index].patient_id)
        val_patients = set(bags.iloc[val_index].patient_id)
        if train_patients & val_patients:
            raise RuntimeError("Patient overlap between train and validation")

        # Expand patient assignments back to the manifest's individual image rows.
        train = manifest.loc[manifest.patient_id.isin(train_patients)].copy()
        val = manifest.loc[manifest.patient_id.isin(val_patients)].copy()
        if set(train.label) != set(bags.label) or set(val.label) != set(bags.label):
            raise ValueError("Grouped stratification produced a fold missing a class; reduce folds")
        folds.append((train, val))

    return folds


def classification_metrics(labels, probabilities):
    # Binary predictions store P(positive); multiclass predictions store one probability per class.
    binary = probabilities.shape[1] == 1
    predicted = (probabilities[:, 0] >= 0.5).astype(int) if binary else probabilities.argmax(1)
    classes = list(range(2 if binary else probabilities.shape[1]))
    average = "binary" if binary else "macro"

    # Macro averaging gives each multiclass label equal weight.
    result = {
        "accuracy": float(metrics.accuracy_score(labels, predicted)),
        "precision": float(metrics.precision_score(labels, predicted, average=average,
                                                  labels=classes, zero_division=0)),
        "sensitivity": float(metrics.recall_score(labels, predicted, average=average,
                                                  labels=classes, zero_division=0)),
        "f1score": float(metrics.f1_score(labels, predicted, average=average,
                                         labels=classes, zero_division=0)),
    }

    # A class-incomplete subset has no defined AUROC; NaN must not be reported as zero.
    if set(labels) != set(classes):
        warnings.warn("AUROC is undefined because this evaluation is missing a class")
        result["roc_auc"] = float("nan")
    elif binary:
        result["roc_auc"] = float(metrics.roc_auc_score(labels, probabilities[:, 0]))
    else:
        result["roc_auc"] = float(metrics.roc_auc_score(labels, probabilities,
                                                       labels=classes, multi_class="ovr"))

    if binary:
        result["specificity"] = float(metrics.recall_score(labels, predicted, pos_label=0,
                                                          zero_division=0))

    return result


def save_checkpoint(path, model, metadata):
    # Serialization finishes before the next training step; no persistent weight copy is needed.
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({**metadata, "model_state_dict": state}, path)


def load_pretraining(model, checkpoint, cfg, train_patients, val_patients):
    # Legacy checkpoints have no trustworthy split provenance and are deliberately rejected.
    expected = {"format_version": 2, "task": cfg["task"], "mil": False,
                "encoder_name": cfg["encoder_kwargs"]["model_name"], "num_out": model.num_out,
                "train_patients": sorted(train_patients), "val_patients": sorted(val_patients)}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Pretraining checkpoint has incompatible {key}; regenerate it")

    # Version-2 instance checkpoints from this workflow already inherited bag labels.
    if checkpoint.get("instance_label_source", "bag") != "bag":
        raise ValueError("Instance pretraining must use bag labels as instance labels")

    # Transfer by named module prefixes, not by the order of checkpoint dictionary entries.
    state = normalize_model_state_dict(checkpoint["model_state_dict"])
    for name, enabled in (("encoder", True), ("clf", cfg.get("instance_load_clf", False))):
        if enabled:
            prefix = name + "."
            weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            getattr(model, name).load_state_dict(weights, strict=True)
