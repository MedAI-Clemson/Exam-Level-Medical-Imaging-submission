"""Validate a YAML configuration before opening data or creating models."""

from pathlib import Path

import yaml

from models import POOLING_METHODS
from pretraining import configure_pretraining, normalize_classifier_options
from training_options import configure_model_output


TASKS = ("achilles", "ct", "mri", "xray")
PATH_KEYS = ("save_dir", "data_dir", "metadata_csv", "targets_csv", "pretrained_model_dir", "radimagenet_checkpoint")


def load_config(path, task=None, overrides=None, net_overrides=None):
    # Read YAML first, then apply command-line overrides before validating settings.
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")

    # Resolve old spellings before CLI overrides, so explicit overrides still take precedence.
    normalize_classifier_options(cfg)
    cfg.update({key: value for key, value in (overrides or {}).items() if value is not None})
    cfg.setdefault("net_kwargs", {}).update(
        {key: value for key, value in (net_overrides or {}).items() if value is not None})

    # A task-specific launcher must agree with the dataset named in the YAML.
    if task is not None and cfg.get("task", task) != task:
        raise ValueError("The launcher task disagrees with the config task")
    cfg["task"] = task or cfg.get("task")
    if cfg["task"] not in TASKS:
        raise ValueError(f"Set task to one of {TASKS}")

    # Shared training settings and defaults.
    for key in ("save_dir", "data_dir", "metadata_csv", "encoder_kwargs", "loader_kwargs",
                "optim_kwargs", "sched_kwargs", "mil", "num_epochs", "folds"):
        if key not in cfg:
            raise ValueError(f"Missing configuration field: {key}")

    cfg.setdefault("seed", 123)
    cfg.setdefault("device", "auto")
    cfg.setdefault("res", 224)
    configure_model_output(cfg)
    if type(cfg["mil"]) is not bool:
        raise ValueError("mil must be true or false")
    for key in ("num_epochs", "folds", "res"):
        if type(cfg[key]) is not int or cfg[key] < (2 if key == "folds" else 1):
            raise ValueError(f"Invalid {key}")

    # The trainer owns shuffling and collation; loader settings only control batching.
    if cfg["loader_kwargs"].get("batch_size", 1) < 1 or cfg["loader_kwargs"].get("num_workers", 0) < 0:
        raise ValueError("Invalid batch_size or num_workers")
    if set(cfg["loader_kwargs"]) - {"batch_size", "num_workers", "pin_memory"}:
        raise ValueError("loader_kwargs supports batch_size, num_workers, and pin_memory")

    # Binary tasks use one output logit; X-ray uses one logit for each of three classes.
    expected_outputs = 3 if cfg["task"] == "xray" else 1
    if cfg["encoder_kwargs"].get("num_classes") != expected_outputs:
        raise ValueError(f"{cfg['task']} requires num_classes={expected_outputs}")
    if cfg["encoder_kwargs"].get("in_chans", 1) != 1:
        raise ValueError("This workflow loads grayscale images and requires in_chans=1")
    cfg["encoder_kwargs"].setdefault("in_chans", 1)

    # Check pooling settings before creating the encoder or downloading its weights.
    options = cfg.setdefault("net_kwargs", {})
    options.setdefault("pooling_method", "attn")
    if type(options.get("num_heads", 1)) is not int or options.get("num_heads", 1) < 1:
        raise ValueError("num_heads must be a positive integer")
    if options["pooling_method"] not in POOLING_METHODS:
        raise ValueError(f"Unknown pooling_method: {options['pooling_method']}")
    if options.get("num_out", expected_outputs) != expected_outputs:
        raise ValueError("num_out and num_classes must agree")
    if options["pooling_method"] == "transformer" and 64 % options.get("num_heads", 1):
        raise ValueError("Transformer num_heads must divide 64")

    # Dataset-specific inputs and legacy settings that must not be silently ignored.
    if cfg.get("view_filter") and cfg["task"] not in ("xray", "mri"):
        raise ValueError("view_filter is supported for xray and mri")
    if cfg["task"] == "xray" and "targets_csv" not in cfg:
        raise ValueError("X-ray configuration requires targets_csv")
    for obsolete in ("balance", "dp", "vid_enc", "pretrain"):
        if cfg.get(obsolete):
            raise ValueError(f"{obsolete} is not supported; use the documented configuration")

    # MRI annotations must explicitly identify how their k coordinate indexes slices.
    if cfg.get("use_k"):
        if cfg["task"] != "mri" or cfg.get("k_index_base") not in (0, 1):
            raise ValueError("use_k requires MRI and an explicit k_index_base of 0 or 1")
        if cfg.get("slice_order") not in ("position", "instance"):
            raise ValueError("use_k requires an explicit slice_order matching the annotation convention")

    # CT uses fixed intensity bounds; MRI uses percentiles of each slice.
    preprocessing = cfg.setdefault("preprocessing", {})
    if cfg["task"] == "ct":
        window = preprocessing.get("ct_window", [])
        if len(window) != 2 or not window[0] < window[1]:
            raise ValueError("Set preprocessing.ct_window to [lower_HU, upper_HU]")
    if cfg["task"] == "mri":
        bounds = preprocessing.setdefault("mri_percentiles", [1, 99])
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 100:
            raise ValueError("Invalid mri_percentiles")

    # Resolve the single encoder weight source and validate transfer/freezing settings.
    configure_pretraining(cfg)

    # Preserve original example locations, but resolve independently of the caller's cwd.
    root = Path(__file__).resolve().parent
    base = (Path(path).resolve().parent / cfg["path_root"]).resolve() if "path_root" in cfg else root
    for key in PATH_KEYS:
        if cfg.get(key):
            cfg[key] = str((base / cfg[key]).resolve())

    return cfg
