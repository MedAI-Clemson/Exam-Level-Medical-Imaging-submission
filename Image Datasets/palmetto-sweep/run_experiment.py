"""Prepare one experiment and run the existing trainer in a fresh attempt directory."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import yaml


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def experiment_config(sweep, dataset, stage, source, pooling, threads, encoder_state=None):
    if stage == "mil":
        if encoder_state not in ("trainable", "frozen"):
            raise ValueError("MIL requires an explicit encoder_state: trainable or frozen")
        if encoder_state not in sweep["encoder_states"]:
            raise ValueError("Requested encoder state is not enabled in the sweep")
    project = Path(sweep["project_dir"]).expanduser().resolve()
    base_path = project / "config" / f"{dataset}_config.yaml"
    cfg = read_yaml(base_path)
    if cfg.get("task") != dataset:
        raise ValueError(f"Task mismatch in {base_path}")

    # Both stages inherit identical data, filters, preprocessing, and patient splits.
    # Only optimization settings and instance batch size come from the old PT config.
    if stage == "instance":
        pt = read_yaml(project / "config" / f"{dataset}_pt_config.yaml")
        for key in ("num_epochs", "optim_kwargs", "sched_kwargs", "loader_kwargs"):
            cfg[key] = pt[key]
        source = sweep["instance_initialization"]

    # One setting can shorten every experiment without editing the dataset configs.
    epochs = sweep.get("num_epochs")
    if epochs is not None:
        if type(epochs) is not int or epochs < 1:
            raise ValueError("num_epochs must be a positive integer or null")
        cfg["num_epochs"] = epochs

    # Resolve dataset paths before moving the generated YAML outside the source tree.
    base = (base_path.parent / cfg["path_root"]).resolve() if "path_root" in cfg else project
    for key in ("data_dir", "metadata_csv", "targets_csv"):
        if cfg.get(key):
            cfg[key] = str((base / cfg[key]).resolve())
    cfg.pop("path_root", None)

    # Drop historical selectors so they cannot override the explicit experiment design.
    for key in ("pretrain_enc", "pretrain_fc", "instance_load_head", "freeze_fc", "pretrained_model_dir"):
        cfg.pop(key, None)
    cfg.update(task=dataset, mil=stage == "mil", pretraining=source, device="cuda",
               seed=sweep["seed"], folds=sweep["folds"], instance_load_clf=False,
               freeze_enc=stage == "mil" and encoder_state == "frozen", freeze_clf=False)
    cfg["model_selection"] = sweep.get("model_selection", "last")
    # Instance checkpoints are required inputs for the downstream MIL jobs.
    cfg["save_models"] = True if stage == "instance" else sweep.get("save_models", True)
    cfg["encoder_kwargs"].update(model_name="resnet50")
    cfg["encoder_kwargs"].pop("pretrained", None)
    cfg.setdefault("net_kwargs", {}).update(pooling_method=pooling, num_heads=sweep["num_heads"])
    cfg["loader_kwargs"]["num_workers"] = max(0, threads - 1)
    if source == "radimagenet":
        cfg["radimagenet_checkpoint"] = str(Path(sweep["radimagenet_checkpoint"]).expanduser().resolve())
    else:
        cfg.pop("radimagenet_checkpoint", None)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--dataset", required=True, choices=("achilles", "ct", "mri", "xray"))
    parser.add_argument("--stage", required=True, choices=("instance", "mil"))
    parser.add_argument("--source", default="imagenet", choices=("imagenet", "radimagenet", "instance"))
    parser.add_argument("--pooling", default="attn")
    parser.add_argument("--encoder-state", choices=("trainable", "frozen"))
    parser.add_argument("--instance-marker", default="")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    sweep = read_yaml(args.sweep)
    cfg = experiment_config(sweep, args.dataset, args.stage, args.source, args.pooling, args.threads, args.encoder_state)

    if cfg["pretraining"] == "instance":
        previous = json.loads(Path(args.instance_marker).read_text(encoding="utf-8"))
        cfg["pretrained_model_dir"] = previous["output_dir"]
        for fold in range(1, cfg["folds"] + 1):
            if not (Path(previous["output_dir"]) / f"model_fold{fold}.ckpt").is_file():
                raise FileNotFoundError(f"Instance checkpoint missing for fold {fold}")

    # A retry starts clean without deleting previous checkpoints or partial results.
    # The trainer cannot resume epochs and refuses a nonempty save_dir.
    label = "instance_pretrain" if args.stage == "instance" else f"{args.source}_{args.pooling}"
    root = Path(sweep["results_dir"]).expanduser().resolve() / sweep["run_name"]
    # Instance pretraining has no encoder-state dimension and is shared by both MIL states.
    state_dir = "shared" if args.stage == "instance" else args.encoder_state
    attempts = root / "attempts" / state_dir / args.dataset / label
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="run_", dir=attempts))
    cfg["save_dir"] = str(attempt / "artifacts")
    generated = attempt / "experiment.yaml"
    generated.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    project = Path(sweep["project_dir"]).expanduser().resolve()
    command = [sweep["python"], str(project / "train.py"), "--config", str(generated)]
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\nAttempt: {attempt}\n")
        stream.flush()
        subprocess.run(command, cwd=project, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    output = Path(cfg["save_dir"])
    required = ["summary.csv", "predictions.csv"]
    if cfg["save_models"]:
        required += [f"model_fold{fold}.ckpt" for fold in range(1, cfg["folds"] + 1)]
    for filename in required:
        if not (output / filename).is_file():
            raise FileNotFoundError(output / filename)

    # Snakemake tracks this small manifest; it is written only after complete success.
    marker = Path(args.marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    record = dict(dataset=args.dataset, stage=args.stage, pretraining=cfg["pretraining"],
                  encoder_state=args.encoder_state if args.stage == "mil" else None,
                  freeze_enc=cfg["freeze_enc"],
                  pooling=args.pooling if args.stage == "mil" else None,
                  output_dir=str(output), config=str(generated), folds=cfg["folds"],
                  model_selection=cfg["model_selection"], save_models=cfg["save_models"])
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(marker)


if __name__ == "__main__":
    main()
