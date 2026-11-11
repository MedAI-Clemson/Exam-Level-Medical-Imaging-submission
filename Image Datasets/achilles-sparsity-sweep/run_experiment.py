"""Prepare Achilles bags, train ten folds per job, and summarize held-out predictions."""

import argparse
import json
import random
import sys
from pathlib import Path

from bags import design_settings, digest, prepare, validate, write_csv, write_json
from metrics import evaluate, fold_summary


def train(cfg, root, args):
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    sys.path.insert(0, str(Path(cfg['project_dir']).expanduser().resolve()))
    from models import build_model
    from baselines import baseline_batch

    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a GPU; submit through the Palmetto profile')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    design_path = root / 'design.json'
    design = json.loads(design_path.read_text(encoding='utf-8'))
    if design['configuration'] != design_settings(cfg) or design['metadata_sha256'] != digest(cfg['metadata_csv']):
        raise ValueError('Bag design does not match this configuration/metadata; rerun preparation')
    run = root / 'runs' / f'n{args.size}' / args.pooling
    marker = root / 'completed' / f'n{args.size}' / f'{args.pooling}.json'
    marker.unlink(missing_ok=True)
    heads = cfg['num_heads_by_pooling'].get(args.pooling, cfg['num_heads'])
    model_cfg = dict(mil=args.pooling != 'instance', pretraining='imagenet',
                     encoder_kwargs=dict(model_name=cfg['encoder'], in_chans=3, num_classes=1, drop_rate=cfg['drop_rate']),
                     net_kwargs=dict(pooling_method=args.pooling, num_heads=heads, drop_rate=cfg['drop_rate']))
    write_json(run / 'config.resolved.json', dict(cfg, bag_size=args.size, method=args.pooling,
                                                model=model_cfg, design_sha256=digest(design_path)))

    # Match the latest frozen MNIST experiment: grayscale repeated to RGB, ImageNet
    # normalization, and no stochastic augmentation. Bag membership is also fixed.
    transform = transforms.Compose([transforms.Resize((cfg['resolution'], cfg['resolution'])),
                                    transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((.485, .456, .406), (.229, .224, .225))])

    class BagDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            images = []
            for image in row['images']:
                with Image.open(image['path']) as opened:
                    images.append(transform(opened.convert('RGB')))
            return torch.stack(images), row['label'], index

    all_predictions, per_fold = [], []
    for fold in range(1, cfg['folds'] + 1):
        seed = cfg['seed'] + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        condition = design['designs'][str(fold)][str(args.size)]
        loaders = {split: DataLoader(BagDataset(rows), batch_size=cfg['batch_size'], shuffle=split == 'train',
                                    num_workers=min(cfg['num_workers'], max(0, args.threads - 1)), pin_memory=True,
                                    generator=torch.Generator().manual_seed(seed))
                   for split, rows in condition.items()}
        model = build_model(model_cfg).cuda()
        model.set_encoder_frozen(cfg['freeze_encoder'])
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                     lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['scheduler_step'], gamma=cfg['scheduler_gamma'])
        criterion = torch.nn.BCEWithLogitsLoss()

        def epoch(split, updating):
            model.train(updating)
            predictions, total_loss = [], 0.0
            with torch.set_grad_enabled(updating):
                for images, labels, indices in loaders[split]:
                    batch, length = images.shape[:2]
                    images = images.flatten(0, 1).cuda(non_blocking=True)
                    labels = labels.float().cuda(non_blocking=True)
                    if updating:
                        optimizer.zero_grad(set_to_none=True)
                    if args.pooling in ('instance', 'stan'):
                        loss, probabilities, _ = baseline_batch(model, images, [length] * batch,
                                                               labels, args.pooling, updating)
                        probabilities = probabilities[:, 0]
                    else:
                        logits = model(images, [length] * batch).flatten()
                        loss = criterion(logits, labels)
                        probabilities = logits.sigmoid()
                    if not torch.isfinite(loss):
                        raise ValueError(f'Non-finite loss in fold {fold}')
                    if updating:
                        loss.backward()
                        optimizer.step()
                    total_loss += loss.item() * batch
                    for index, probability in zip(indices.tolist(), probabilities.detach().cpu().tolist()):
                        row = condition[split][index]
                        predictions.append(dict(patient_id=row['patient_id'], fold=fold, bag_size=args.size,
                                                pooling=args.pooling, label=row['label'], severity=row['severity'],
                                                source_count=row['source_count'], source_fraction=row['source_count']/args.size,
                                                probability=probability, predicted_label=int(probability >= cfg['threshold'])))
            return dict(loss=total_loss/len(predictions), **evaluate(predictions, cfg['threshold'])), predictions

        history = []
        for number in range(1, cfg['num_epochs'] + 1):
            training, _ = epoch('train', True)
            validation, predictions = epoch('val', False)
            history.append(dict(epoch=number, learning_rate=optimizer.param_groups[0]['lr'],
                                **{f'train_{k}': v for k, v in training.items()},
                                **{f'val_{k}': v for k, v in validation.items()}))
            write_csv(run / f'fold{fold}' / 'history.csv', history)
            print(f'Fold {fold}, epoch {number}: validation {validation}', flush=True)
            scheduler.step()

        # Only last-epoch validation predictions contribute to the final evaluation.
        write_csv(run / f'fold{fold}' / 'predictions.csv', predictions)
        all_predictions.extend(predictions)
        per_fold.append(dict(fold=fold, bag_size=args.size, pooling=args.pooling, **validation))
        del model, optimizer, scheduler, loaders
        torch.cuda.empty_cache()

    expected = set(design['folds'])
    if len(all_predictions) != len(expected) or {r['patient_id'] for r in all_predictions} != expected:
        raise ValueError('Each patient must have exactly one held-out prediction')
    write_csv(run / 'predictions.csv', all_predictions)
    write_csv(run / 'fold_metrics.csv', per_fold)
    result = dict(bag_size=args.size, pooling=args.pooling, encoder_state='frozen' if cfg['freeze_encoder'] else 'trainable',
                  pretraining='imagenet', replacement_if_needed=cfg.get('replacement_if_needed', False), num_heads=heads, folds=10, num_epochs=cfg['num_epochs'], threshold=cfg['threshold'],
                  model_selection='last', save_models=False, **fold_summary(per_fold),
                  **{f'oof_{k}': v for k, v in evaluate(all_predictions, cfg['threshold']).items()},
                  design_sha256=digest(design_path), run_dir=str(run))
    write_json(run / 'summary.json', result)
    write_json(marker, result)
    print('Completed all ten folds.', flush=True)


def summarize(cfg, root):
    import csv
    summary, folds, predictions = [], [], []
    for size in cfg['bag_sizes']:
        for method in cfg['pooling']:
            marker = root / 'completed' / f'n{size}' / f'{method}.json'
            result = json.loads(marker.read_text(encoding='utf-8'))
            if result['design_sha256'] != digest(root / 'design.json'):
                raise ValueError(f'Stale result: {marker}')
            summary.append(result)
            run = Path(result['run_dir'])
            for name, target in [('fold_metrics.csv', folds), ('predictions.csv', predictions)]:
                with (run / name).open(newline='', encoding='utf-8') as stream:
                    target.extend(csv.DictReader(stream))
    write_csv(root / 'summary.csv', summary)
    write_csv(root / 'fold_metrics.csv', folds)
    write_csv(root / 'predictions.csv', predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'summarize'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--pooling')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    path = Path(args.config)
    if path.suffix == '.json':
        cfg = json.loads(path.read_text(encoding='utf-8'))
    else:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    validate(cfg)
    root = Path(cfg['results_dir']).expanduser().resolve() / cfg['run_name']
    if args.action == 'prepare':
        prepare(cfg, root)
        import timm
        timm.create_model(cfg['encoder'], pretrained=True, num_classes=0)
    elif args.action == 'train':
        if args.size not in cfg['bag_sizes'] or args.pooling not in cfg['pooling']:
            parser.error('Requested method/bag size is not enabled')
        train(cfg, root, args)
    else:
        summarize(cfg, root)


if __name__ == '__main__':
    main()
