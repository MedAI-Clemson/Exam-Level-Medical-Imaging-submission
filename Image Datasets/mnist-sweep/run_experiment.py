"""Prepare MNIST bags, train the existing MIL models, and collect completed runs."""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

from bags import make_bags, source_split, validate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def prepare(cfg, root):
    from torchvision.datasets import MNIST
    import timm

    cache = str(Path(cfg['data_dir']).expanduser().resolve())
    training = MNIST(cache, train=True, download=True)
    testing = MNIST(cache, train=False, download=True)
    labels = training.targets.tolist()
    train_ids, val_ids = source_split(labels, cfg['validation_fraction'], cfg['seed'])
    splits = {'train': (labels, train_ids), 'val': (labels, val_ids),
              'test': (testing.targets.tolist(), list(range(len(testing))))}
    design = dict(seed=cfg['seed'], target_digit=cfg['target_digit'],
                  source_indices={'train': train_ids, 'val': val_ids, 'test': splits['test'][1]}, conditions={})
    for size in cfg['bag_sizes']:
        for density in cfg['densities']:
            condition = {}
            for offset, (split, (digit_labels, ids)) in enumerate(splits.items()):
                condition[split] = make_bags(digit_labels, ids, cfg[f'{split}_bags'], size, density,
                                             cfg['target_digit'], cfg['seed'] + offset * 100000 + size * 100 + density)
            design['conditions'][f'n{size}/p{density}'] = condition
    # Download encoder weights once. GPU jobs use the shared user's cache.
    timm.create_model(cfg['encoder'], pretrained=True, num_classes=0)
    write_json(root / 'design.json', design)
    print('Prepared all shared bag manifests and cached ImageNet weights.', flush=True)


def train(cfg, root, args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from sklearn.metrics import roc_auc_score, brier_score_loss

    sys.path.insert(0, str(Path(cfg['project_dir']).expanduser().resolve()))
    from models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a GPU; run the preparation/dry-run locally only.')
    torch.set_num_threads(args.threads)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    design_path = root / 'design.json'
    design = json.loads(design_path.read_text())
    condition = design['conditions'][f'n{args.size}/p{args.density}']
    run = root / 'runs' / f'n{args.size}' / f'p{args.density}' / args.pooling
    run.mkdir(parents=True, exist_ok=True)
    marker = root / 'completed' / f'n{args.size}' / f'p{args.density}' / f'{args.pooling}.json'
    # An interrupted/restarted attempt must not retain a success marker.
    marker.unlink(missing_ok=True)
    model_cfg = dict(mil=True, pretraining='imagenet',
                     encoder_kwargs=dict(model_name=cfg['encoder'], in_chans=3, num_classes=1, drop_rate=cfg['drop_rate']),
                     net_kwargs=dict(pooling_method=args.pooling, num_heads=cfg['num_heads'], drop_rate=cfg['drop_rate']))
    resolved = dict(cfg, bag_size=args.size, density_percent=args.density, pooling_method=args.pooling,
                    freeze_encoder=False, model=model_cfg, design_sha256=hashlib.sha256(design_path.read_bytes()).hexdigest())
    write_json(run / 'config.resolved.json', resolved)

    transform = transforms.Compose([transforms.Resize((cfg['resolution'], cfg['resolution'])),
                                    transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    cache = str(Path(cfg['data_dir']).expanduser().resolve())
    sources = {True: MNIST(cache, train=True, download=False), False: MNIST(cache, train=False, download=False)}

    class BagDataset(Dataset):
        def __init__(self, split):
            self.rows = condition[split]
            self.images = sources[split != 'test']

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            images = torch.stack([transform(self.images[i][0]) for i in row['indices']])
            return images, row['label'], row['bag_id'], row['target_count']

    # Fixed-size bags permit the default collator. Flatten only the image/bag axes.
    loaders = {split: DataLoader(BagDataset(split), batch_size=cfg['batch_size'], shuffle=split == 'train',
                                num_workers=cfg['num_workers'], pin_memory=True,
                                generator=torch.Generator().manual_seed(cfg['seed'])) for split in condition}
    model = build_model(model_cfg).cuda()
    model.set_encoder_frozen(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['scheduler_step'], gamma=cfg['scheduler_gamma'])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def epoch(split, updating=False):
        model.train(updating)
        records, total_loss = [], 0.0
        with torch.set_grad_enabled(updating):
            for images, labels, ids, counts in loaders[split]:
                batch, length = images.shape[:2]
                images = images.flatten(0, 1).cuda(non_blocking=True)
                target = labels.float().cuda(non_blocking=True)
                if updating:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images, [length] * batch).flatten()
                loss = loss_fn(logits, target)
                if updating:
                    loss.backward(); optimizer.step()
                total_loss += loss.item() * batch
                probabilities = logits.detach().sigmoid().cpu().tolist()
                records.extend(dict(bag_id=int(i), label=int(y), probability=float(p), target_count=int(k),
                                    bag_size=length) for i, y, p, k in zip(ids, labels, probabilities, counts))
        y = [r['label'] for r in records]; p = [r['probability'] for r in records]
        return dict(loss=total_loss / len(records), auroc=float(roc_auc_score(y, p)),
                    brier=float(brier_score_loss(y, p))), records

    history = []
    for number in range(1, cfg['num_epochs'] + 1):
        training, _ = epoch('train', updating=True)
        validation, _ = epoch('val')
        history.append(dict(epoch=number, learning_rate=optimizer.param_groups[0]['lr'],
                            **{f'train_{k}': v for k, v in training.items()},
                            **{f'val_{k}': v for k, v in validation.items()}))
        write_csv(run / 'history.csv', history)
        print(f'Epoch {number}/{cfg["num_epochs"]}: train {training}, validation {validation}', flush=True)
        scheduler.step()

    # Test only after the final epoch; no test-driven checkpoint selection.
    metrics, predictions = epoch('test')
    write_csv(run / 'test_predictions.csv', predictions)
    result = dict(bag_size=args.size, density_percent=args.density, pooling=args.pooling,
                  seed=cfg['seed'], epochs=cfg['num_epochs'], pretraining='imagenet', encoder_state='trainable',
                  model_selection='last', save_models=False, test_bags=len(predictions),
                  **metrics, run_dir=str(run), design_sha256=resolved['design_sha256'])
    write_json(run / 'summary.json', result)
    write_json(marker, result)
    print('Completed:', result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'summarize'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--size', type=int)
    parser.add_argument('--density', type=int)
    parser.add_argument('--pooling')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    validate(cfg)
    root = Path(cfg['results_dir']).expanduser().resolve() / cfg['run_name']
    if args.action == 'prepare':
        prepare(cfg, root)
    elif args.action == 'train':
        if args.size not in cfg['bag_sizes'] or args.density not in cfg['densities'] or args.pooling not in cfg['pooling']:
            parser.error('Training condition is not in the sweep')
        train(cfg, root, args)
    else:
        rows = [json.loads((root / 'completed' / f'n{n}' / f'p{d}' / f'{p}.json').read_text())
                for n in cfg['bag_sizes'] for d in cfg['densities'] for p in cfg['pooling']]
        write_csv(root / 'summary.csv', rows)


if __name__ == '__main__':
    main()
