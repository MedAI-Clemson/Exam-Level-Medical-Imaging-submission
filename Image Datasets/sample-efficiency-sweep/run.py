"""Prepare, train, and summarize the independent sample-efficiency experiment."""

import argparse
import copy
import csv
import json
import hashlib
import subprocess
import sys
from pathlib import Path

from support import POOLING, COMPARATORS, BASELINES, SinglePassBatchSampler, conditions, immutable_json, prepare, resolve_config


def locations(root, condition, method, fold):
    return (root / 'runs' / condition / method / f'fold{fold}',
            root / 'completed' / condition / method / f'fold{fold}.json')


def training_signature(cfg, root):
    """Invalidate resumable folds when settings, prepared data, or training code change."""
    project = Path(cfg['project_dir'])
    files = [root / 'manifest.json']
    for folder in (Path(__file__).parent, project / 'comparison-sweep'):
        files.extend(p for p in folder.glob('*.py') if not p.name.startswith('test_'))
    files.extend(project / n for n in ('train.py', 'configuration.py', 'datasets.py', 'models.py',
                 'pretraining.py', 'training_options.py', 'utils.py', 'imaging.py', 'transforms.py', 'baselines.py'))
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for path in sorted(files):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fold_receipt(directory, signature):
    return dict(signature=signature, files={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ('predictions.csv', 'fold_metrics.csv', 'split.json')})


def train_all(cfg, root, name, method):
    """Run ten folds sequentially; keep completed folds through job failure or timeout."""
    signature = training_signature(cfg, root)
    if method in ('dtfd_afs', 'clam'):
        dataset = conditions(cfg)[name]['dataset']
        receipt = root / 'features' / dataset / 'complete.json'
        signature = hashlib.sha256(signature.encode() + receipt.read_bytes()).hexdigest()
    complete = root / 'completed' / name / method / 'complete.json'
    complete.unlink(missing_ok=True)
    for number in range(1, 11):
        directory, marker = locations(root, name, method, number)
        try:
            saved = json.loads(marker.read_text(encoding='utf-8'))
            reusable = saved.get('resume') == fold_receipt(directory, signature)
        except (OSError, ValueError):
            reusable = False
        if reusable:
            print(f'Fold {number}/10: already complete; skipping.', flush=True)
            continue
        marker.unlink(missing_ok=True)
        log = root / 'logs' / name / method / f'fold{number}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f'Fold {number}/10: training; log: {log}', flush=True)
        # A fresh process releases all GPU memory between folds.
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), 'train',
                            '--config', str(root / 'sweep.resolved.json'), '--condition', name,
                            '--method', method, '--fold', str(number)], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        saved = json.loads(marker.read_text(encoding='utf-8'))
        saved['resume'] = fold_receipt(directory, signature)
        marker.write_text(json.dumps(saved, indent=2), encoding='utf-8')

    # Only combined outputs belong to Snakemake; failed-job cleanup preserves fold files.
    target = root / 'runs' / name / method
    for filename in ('predictions.csv', 'fold_metrics.csv'):
        rows = []
        for number in range(1, 11):
            directory, _ = locations(root, name, method, number)
            with (directory / filename).open(newline='', encoding='utf-8') as stream:
                rows.extend(csv.DictReader(stream))
        with (target / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    complete.write_text(json.dumps(dict(condition=name, method=method, folds=10, signature=signature), indent=2), encoding='utf-8')


def train_pooling(cfg, root, name, method, number):
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from datasets import ImagingDataset, collate_bags
    from design import digest, write_csv, write_json
    from models import build_model
    from train import run_epoch
    from train_comparison import metrics
    from utils import seed_everything, seed_worker

    if not torch.cuda.is_available():
        raise RuntimeError('Submit training on a GPU through the Slurm profile')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    condition = manifest['conditions'][name]
    dataset = manifest['datasets'][condition['dataset']]
    split = condition['folds'][number-1]
    model_cfg = copy.deepcopy(manifest['pooling_configs'][condition['task']])
    model_cfg['net_kwargs']['pooling_method'] = method
    model_cfg['num_epochs'] = cfg.get('epochs_override') or model_cfg['num_epochs']
    model_cfg['loader_kwargs']['num_workers'] = cfg['num_workers']
    directory, marker = locations(root, name, method, number)
    marker.unlink(missing_ok=True)
    model_cfg['save_dir'] = str(directory)
    write_json(directory / 'config.resolved.json', dict(model_cfg, mil=method != 'instance',
               baseline=method if method in BASELINES else None))

    # Matched initializations for all fractions; subset sampling has its own seed.
    seed = cfg['seed'] + number
    seed_everything(seed)
    torch.set_num_threads(cfg['cpus'])
    encoder_cfg = dict(model_cfg, mil=method != 'instance')
    model = build_model(encoder_cfg)
    model.set_encoder_frozen(True)
    model.clf.requires_grad_(True)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.clf.parameters())
    model.to('cuda')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], **model_cfg['optim_kwargs'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **model_cfg['sched_kwargs'])

    frames = {}
    for part in ('train', 'val'):
        frames[part] = pd.DataFrame([dict(bag_id=b['bag_id'], patient_id=b['patient_id'], label=b['label'],
                                         path=dataset['images'][image]['path'])
                                    for b in split[part] for image in b['images']])
        frames[part].to_csv(directory / f'{part}.csv', index=False)
    train_data = ImagingDataset(frames['train'], model_cfg, train=True)
    val_data = ImagingDataset(frames['val'], model_cfg, train=False)
    bag_budget = min(len(train_data), cfg.get('smoke_bags_per_epoch') or len(train_data))
    batch_size = model_cfg['loader_kwargs']['batch_size']
    sampler = SinglePassBatchSampler(len(train_data), batch_size, bag_budget, seed)
    common = dict(num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_bags,
                  worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, batch_sampler=sampler,
                              generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(seed+1), **common)
    epoch_runner = run_epoch
    if method in BASELINES:
        from functools import partial
        from baselines import run_baseline_epoch
        epoch_runner = partial(run_baseline_epoch, method=method)
    history = []
    for epoch in range(1, model_cfg['num_epochs']+1):
        training, _ = epoch_runner(model, train_loader, 'cuda', optimizer)
        validation, predictions = epoch_runner(model, val_loader, 'cuda')
        scheduler.step()
        history.append(dict(epoch=epoch, optimizer_updates=epoch*len(sampler),
                            train_loss=training['loss'], val_loss=validation['loss']))
        write_csv(directory / 'history.csv', history)
        print(f'{name} {method} fold {number} epoch {epoch}: '
              f'train loss {training["loss"]:.4f}; val loss {validation["loss"]:.4f}', flush=True)
    if model.num_out == 1:
        predictions['probability_0'] = 1-predictions['probability_1']
    records = predictions.to_dict('records')
    for r in records:
        r.update(fold=number, selected_epoch=epoch, condition=name, method=method)
    score = dict(fold=number, selected_epoch=epoch, loss=validation['loss'],
                 optimizer_updates=epoch*len(sampler), bags_per_reference_epoch=bag_budget,
                 **metrics(records, dataset['num_classes']))
    write_csv(directory / 'predictions.csv', records)
    write_csv(directory / 'fold_metrics.csv', [score])
    write_json(marker, dict(condition=name, method=method, run_dir=str(directory),
                           manifest_sha256=digest(root / 'manifest.json'), model_selection='last',
                           save_models=False, encoder_profile='imagenet_frozen', classifier_initialization='random',
                           classifier_trainable=True, epoch_budget=epoch))


def enrich_result(cfg, root, name, method, number):
    """Same fold output schema for pooling and native comparator training."""
    from design import write_csv, write_json
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    condition = manifest['conditions'][name]
    split = condition['folds'][number-1]
    directory, marker = locations(root, name, method, number)
    result = json.loads(marker.read_text(encoding='utf-8'))
    info = dict(dataset=condition['task'], train_percent=condition['train_percent'],
                sampling_seed=condition['sampling_seed'], fold=number,
                train_patients=len({b['patient_id'] for b in split['train']}),
                train_bags=len(split['train']), full_train_bags=condition['full_train_bags'][number-1],
                val_patients=len({b['patient_id'] for b in split['val']}), val_bags=len(split['val']),
                budget_mode='single_pass', smoke_test=any(cfg.get(k) is not None for k in ('epochs_override', 'smoke_bags_per_epoch')))
    # Completion is restored only after all final artifacts are written successfully.
    marker.unlink()
    for filename in ('fold_metrics.csv', 'predictions.csv'):
        with (directory / filename).open(newline='', encoding='utf-8') as stream:
            rows = [dict(r, **info, condition=name, method=method) for r in csv.DictReader(stream)]
        write_csv(directory / filename, rows)
    write_json(directory / 'split.json', dict(train=split['train'], val=split['val']))
    write_json(marker, dict(result, **info))


def summarize(cfg, root):
    import numpy as np
    from scipy.stats import t
    from design import digest, write_csv
    from train_comparison import metrics
    summaries, all_folds, all_predictions = [], [], []
    manifest_hash = digest(root / 'manifest.json')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, spec in conditions(cfg).items():
        for method in cfg['methods']:
            fold_scores, predictions, markers = [], [], []
            for number in range(1, 11):
                directory, path = locations(root, name, method, number)
                result = json.loads(path.read_text(encoding='utf-8'))
                if result['manifest_sha256'] != manifest_hash or result['fold'] != number:
                    raise ValueError(f'Stale fold completion: {path}')
                markers.append(result)
                with (directory / 'fold_metrics.csv').open(newline='', encoding='utf-8') as stream:
                    scores = list(csv.DictReader(stream))
                if len(scores) != 1 or int(scores[0]['fold']) != number:
                    raise ValueError(f'Invalid fold metrics: {directory}')
                fold_scores.extend(scores)
                with (directory / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    row['label'] = int(row['label'])
                    for key in list(row):
                        if key.startswith('probability_'):
                            row[key] = float(row[key])
                expected = {(b['bag_id'], b['patient_id'], b['label'])
                            for b in manifest['conditions'][name]['folds'][number-1]['val']}
                actual = {(r['bag_id'], r['patient_id'], r['label']) for r in rows}
                if actual != expected or len(rows) != len(expected) or any(int(r['fold']) != number for r in rows):
                    raise ValueError(f'Predictions do not cover the exact held-out fold: {directory}')
                predictions.extend(rows)
            ids = [r['bag_id'] for r in predictions]
            if len(ids) != len(set(ids)):
                raise ValueError('A bag was held out more than once')
            classes = 3 if spec['task'] == 'xray' else 2
            summary = dict(condition=name, method=method, dataset=spec['task'], train_percent=spec['train_percent'],
                           sampling_seed=spec['sampling_seed'], folds=10,
                           encoder_profile=markers[0]['encoder_profile'], model_selection='last', save_models=False,
                           budget_mode='single_pass', smoke_test=markers[0]['smoke_test'],
                           train_patients_mean=float(np.mean([r['train_patients'] for r in markers])),
                           train_patients_min=min(r['train_patients'] for r in markers),
                           train_patients_max=max(r['train_patients'] for r in markers),
                           **{f'pooled_{k}': v for k, v in metrics(predictions, classes).items()})
            for metric in ('auroc', 'brier', 'sensitivity', 'specificity', 'balanced_accuracy'):
                values = np.array([float(r[metric]) for r in fold_scores])
                if not np.isfinite(values).all():
                    raise ValueError(f'Missing {metric} in {name}/{method}')
                mean = float(values.mean())
                margin = float(t.ppf(.95, 9)*values.std(ddof=1)/np.sqrt(10))
                summary.update({f'{metric}_mean':mean, f'{metric}_ci90_low':mean-margin, f'{metric}_ci90_high':mean+margin})
            summaries.append(summary)
            all_folds.extend(fold_scores)
            all_predictions.extend(predictions)
    write_csv(root / 'summary.csv', summaries)
    write_csv(root / 'fold_metrics.csv', all_folds)
    write_csv(root / 'predictions.csv', all_predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'extract', 'train', 'train-all', 'summarize'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--condition')
    parser.add_argument('--method', choices=POOLING + COMPARATORS + BASELINES)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--dataset')
    args = parser.parse_args()
    if args.config.suffix == '.json':
        raw = json.loads(args.config.read_text(encoding='utf-8'))
    else:
        import yaml
        raw = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    cfg = resolve_config(raw, args.config.resolve().parent)
    sys.path.insert(0, cfg['project_dir'])
    sys.path.insert(0, str(Path(cfg['project_dir']) / 'comparison-sweep'))
    root = Path(cfg['results_dir']) / cfg['run_name']
    immutable_json(root / 'sweep.resolved.json', cfg)
    if args.action == 'prepare':
        prepare(cfg, root)
    elif args.action == 'extract':
        from image_data import extract_features
        if args.dataset not in {'medical_'+d for d in cfg['medical_datasets']}:
            parser.error('Dataset is not enabled')
        extract_features(cfg, root, args.dataset)
    elif args.action == 'train-all':
        if args.condition not in conditions(cfg) or args.method not in cfg['methods']:
            parser.error('Choose an enabled condition and method')
        train_all(cfg, root, args.condition, args.method)
    elif args.action == 'train':
        if args.condition not in conditions(cfg) or args.method not in cfg['methods'] or args.fold not in range(1,11):
            parser.error('Choose an enabled condition, method, and fold (1..10)')
        if args.method in POOLING + BASELINES:
            train_pooling(cfg, root, args.condition, args.method, args.fold)
        else:
            from train_comparison import train
            manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
            selected_bags = len(manifest['conditions'][args.condition]['folds'][args.fold-1]['train'])
            budget = min(selected_bags, cfg.get('smoke_bags_per_epoch') or selected_bags)
            train(cfg, root, args.condition, args.method, fold_number=args.fold, training_bags_per_epoch=budget)
        enrich_result(cfg, root, args.condition, args.method, args.fold)
    else:
        summarize(cfg, root)


if __name__ == '__main__':
    main()
