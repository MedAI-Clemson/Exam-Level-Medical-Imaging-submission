"""Entry point for the independent paper-method comparison sweep."""

import argparse
import csv
import json
import sys
from pathlib import Path

from design import METHODS, conditions, digest, prepare, resolve_config, write_csv


def summarize(cfg, root):
    summaries, folds, predictions = [], [], []
    for condition in conditions(cfg):
        for method in cfg['methods']:
            path = root / 'completed' / condition / f'{method}.json'
            result = json.loads(path.read_text(encoding='utf-8'))
            if result['manifest_sha256'] != digest(root / 'manifest.json'):
                raise ValueError(f'Stale result: {path}')
            summaries.append(result)
            for filename, target in [('fold_metrics.csv', folds), ('predictions.csv', predictions)]:
                with (Path(result['run_dir']) / filename).open(newline='', encoding='utf-8') as stream:
                    for row in csv.DictReader(stream):
                        target.append(dict(row, condition=condition, method=method))
    write_csv(root / 'summary.csv', summaries)
    write_csv(root / 'fold_metrics.csv', folds)
    write_csv(root / 'predictions.csv', predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'extract', 'train', 'summarize'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--condition')
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    path = Path(args.config).resolve()
    if path.suffix == '.json':
        cfg = json.loads(path.read_text(encoding='utf-8'))
    else:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg = resolve_config(cfg, path.parent)
    sys.path.insert(0, cfg['project_dir'])
    root = Path(cfg['results_dir']) / cfg['run_name']
    if args.action == 'prepare':
        prepare(cfg, root)
    elif args.action == 'extract':
        from image_data import extract_features
        if args.dataset not in {s['dataset'] for s in conditions(cfg).values()}:
            parser.error('Dataset not enabled')
        extract_features(cfg, root, args.dataset)
    elif args.action == 'train':
        from train_comparison import train
        if args.condition not in conditions(cfg) or args.method not in cfg['methods']:
            parser.error('Condition/method not enabled')
        train(cfg, root, args.condition, args.method)
    else:
        summarize(cfg, root)


if __name__ == '__main__':
    main()
