"""One training loop for all four datasets and both instance/MIL experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from configuration import load_config
from datasets import ImagingDataset, build_manifest, collate_bags
from models import POOLING_METHODS, build_model
from pretraining import PRETRAINING_OPTIONS
from training_options import MODEL_SELECTIONS, FoldSelection, configure_model_output
from utils import (classification_metrics, load_pretraining, make_folds,
                   save_checkpoint, seed_everything, seed_worker)


def run_epoch(model, loader, device, optimizer=None):
    # Passing an optimizer enables training; omitting it runs validation without gradients.
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss() if model.num_out == 1 else nn.CrossEntropyLoss()

    labels_list, probabilities_list, records = [], [], []
    total_loss, count = 0.0, 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            # Keep the sample dimension even when the batch contains only one sample.
            logits = model(images, batch["lengths"])
            if logits.shape != (len(labels), model.num_out):
                raise ValueError(f"Unexpected model output shape: {logits.shape}")

            loss = (criterion(logits[:, 0], labels.float()) if model.num_out == 1
                    else criterion(logits, labels.long()))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; inspect inputs and model outputs")

            if training:
                loss.backward()
                optimizer.step()

            # Metrics use probabilities; losses above use the unnormalized logits.
            probabilities = logits.sigmoid() if model.num_out == 1 else logits.softmax(-1)
            labels_list.append(labels.detach().cpu().numpy())
            probabilities_list.append(probabilities.detach().cpu().numpy())
            records.extend(zip(batch["patient_id"], batch["bag_id"], batch["instance_path"]))

            # Weight by sample count so a short final batch is not overrepresented.
            total_loss += loss.item() * len(labels)
            count += len(labels)

    if not count:
        raise ValueError("Cannot evaluate an empty loader")

    # Calculate metrics over the whole epoch, not an average of per-batch metrics.
    labels, probabilities = np.concatenate(labels_list), np.concatenate(probabilities_list)
    metrics = {"loss": total_loss / count, **classification_metrics(labels, probabilities)}

    # Keep patient/bag IDs attached to the predictions for later inspection.
    predictions = pd.DataFrame(records, columns=["patient_id", "bag_id", "instance_path"])
    predictions["label"] = labels
    for column in range(model.num_out):
        predictions[f"probability_{1 if model.num_out == 1 else column}"] = probabilities[:, column]

    return metrics, predictions


def main(cfg):
    # Run setup: reproducibility, device selection, and patient-isolated folds.
    configure_model_output(cfg)
    seed_everything(cfg["seed"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if cfg["device"] == "auto" else cfg["device"])
    manifest = build_manifest(cfg)
    folds = make_folds(manifest, cfg["folds"], cfg["seed"])

    print(f"Encoder initialization: {cfg['pretraining']} | freeze_enc: {cfg.get('freeze_enc', False)}")
    print(f"Fold model: {cfg['model_selection']} | save_models: {cfg['save_models']}")

    # Save the resolved configuration and exact image list used by this run.
    output = Path(cfg["save_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose --save-dir for a new run.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest.to_csv(output / "manifest.csv", index=False)

    summaries, all_predictions = [], []
    for fold, (train_data, val_data) in enumerate(folds, start=1):
        # Record this fold's split before constructing a fresh model.
        seed_everything(cfg["seed"] + fold)
        train_data.to_csv(output / f"train_fold{fold}.csv", index=False)
        val_data.to_csv(output / f"val_fold{fold}.csv", index=False)
        train_patients, val_patients = sorted(train_data.patient_id.unique()), sorted(val_data.patient_id.unique())

        # Optional transfer is checked against these exact patient partitions.
        model = build_model(cfg)
        if cfg["pretraining"] == "instance":
            checkpoint = torch.load(Path(cfg["pretrained_model_dir"]) / f"model_fold{fold}.ckpt",
                                    map_location="cpu", weights_only=True)
            load_pretraining(model, checkpoint, cfg, train_patients, val_patients)
            del checkpoint

        # Apply freezing before selecting parameters for the optimizer.
        model.set_encoder_frozen(cfg.get("freeze_enc", False))
        if cfg.get("freeze_clf", False):
            model.clf.requires_grad_(False)
        model.to(device)

        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters remain")
        optimizer = torch.optim.Adam(parameters, **cfg["optim_kwargs"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg["sched_kwargs"])

        # Training shuffles and augments; validation keeps a stable order and transform.
        loaders = []
        for training, data in ((True, train_data), (False, val_data)):
            generator = torch.Generator().manual_seed(cfg["seed"] + fold + int(training))
            loaders.append(DataLoader(ImagingDataset(data, cfg, train=training), shuffle=training,
                                      collate_fn=collate_bags, worker_init_fn=seed_worker,
                                      generator=generator, **cfg["loader_kwargs"]))
        train_loader, val_loader = loaders

        # Store split information with weights so later transfer can detect leakage.
        selection = FoldSelection(cfg["model_selection"])
        provenance = {"format_version": 2, "task": cfg["task"], "mil": cfg["mil"],
                      "encoder_name": cfg["encoder_kwargs"]["model_name"], "num_out": model.num_out,
                      "train_patients": train_patients, "val_patients": val_patients, "config": cfg,
                      "pretraining": cfg["pretraining"], "instance_label_source": "bag" if not cfg["mil"] else None}

        # Keep validation outputs from the chosen epoch even when weight saving is disabled.
        selected_metrics, selected_predictions = None, None
        history = []
        for epoch in range(1, cfg["num_epochs"] + 1):
            train_metrics, _ = run_epoch(model, train_loader, device, optimizer)
            val_metrics, val_predictions = run_epoch(model, val_loader, device)
            scheduler.step()

            selected = selection.update(val_metrics["loss"], epoch)
            if selected:
                selected_metrics, selected_predictions = val_metrics, val_predictions

                # Best-loss mode replaces one file on improvement; last mode writes only at the end.
                if cfg["save_models"] and (cfg["model_selection"] == "best_loss" or epoch == cfg["num_epochs"]):
                    save_checkpoint(output / f"model_fold{fold}.ckpt", model,
                                    {**provenance, "epoch": epoch, "loss": val_metrics["loss"],
                                     "metrics": val_metrics, "model_selection": cfg["model_selection"]})

            # Write history each epoch so completed progress remains inspectable.
            history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                            **{f"val_{k}": v for k, v in val_metrics.items()}})
            pd.DataFrame(history).to_csv(output / f"metrics_fold{fold}.csv", index=False)
            print(f"Fold {fold}/{cfg['folds']} | Epoch {epoch}/{cfg['num_epochs']} | "
                  f"train loss {train_metrics['loss']:.4f} | val loss {val_metrics['loss']:.4f}"
                  + (" | selected" if selected else ""), flush=True)

        # These predictions were evaluated with the selected weights at that epoch.
        # No checkpoint reload or extra validation pass is needed.
        selected_predictions.insert(0, "fold", fold)
        selected_predictions.insert(1, "epoch", selection.epoch)
        selected_predictions.to_csv(output / f"preds_fold{fold}.csv", index=False)
        all_predictions.append(selected_predictions)
        summaries.append({"fold": fold, "selected_epoch": selection.epoch,
                          "model_selection": cfg["model_selection"], **selected_metrics})

        # Release this fold's model and optimizer before creating the next one.
        del model, optimizer, scheduler, parameters, train_loader, val_loader, loaders

    # Combine the per-fold results while keeping fold IDs in the prediction table.
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    print(f"Completed {len(folds)} folds. Results: {output}")


def cli(task=None):
    # The same CLI serves train.py and the four task-specific launchers.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--save-dir")
    parser.add_argument("--model-selection", choices=MODEL_SELECTIONS, help="Fold model: last epoch (default) or lowest validation loss")
    parser.add_argument("--save-models", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--pooling-method", choices=POOLING_METHODS)
    parser.add_argument("--pretraining", choices=PRETRAINING_OPTIONS)
    parser.add_argument("--radimagenet-checkpoint", help="Local official PyTorch ResNet-50 weights")
    parser.add_argument("--pretrained-model-dir", help="Directory of matching instance-pretraining fold checkpoints")
    parser.add_argument("--instance-load-clf", action=argparse.BooleanOptionalAction, default=None)
    # Accept the previous CLI spelling without advertising ambiguous terminology in help.
    parser.add_argument("--instance-load-head", action=argparse.BooleanOptionalAction,
                        dest="instance_load_clf", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=None, dest="freeze_enc")
    parser.add_argument("--check-config", action="store_true", help="Validate and print configuration without loading data")
    args = parser.parse_args()

    # Resolve overrides and validate settings before touching any dataset files.
    overrides = {"device": args.device, "save_dir": args.save_dir, "pretraining": args.pretraining,
                 "model_selection": args.model_selection, "save_models": args.save_models,
                 "radimagenet_checkpoint": args.radimagenet_checkpoint,
                 "pretrained_model_dir": args.pretrained_model_dir, "freeze_enc": args.freeze_enc,
                 "instance_load_clf": args.instance_load_clf}

    # Switching away from instance weights also disables that source's optional classifier transfer.
    if args.pretraining in ("imagenet", "radimagenet") and args.instance_load_clf is None:
        overrides["instance_load_clf"] = False
    cfg = load_config(args.config, task, overrides,
                      {"num_heads": args.num_heads, "pooling_method": args.pooling_method})

    if args.check_config:
        print(json.dumps(cfg, indent=2))
    else:
        main(cfg)


if __name__ == "__main__":
    cli()
