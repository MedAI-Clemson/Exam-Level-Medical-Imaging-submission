"""Build comparison jobs and reuse the exact existing sparsity-bag designs."""

import csv
import hashlib
import json
import re
from pathlib import Path


METHODS = ('dtfd_afs', 'clam', 'promil')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return
    tmp = path.with_suffix('.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_config(cfg, base):
    cfg = dict(cfg)
    for key in ('project_dir', 'results_dir', 'mnist_data_dir'):
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((Path(base) / path).resolve())
    if not cfg['methods'] or len(set(cfg['methods'])) != len(cfg['methods']) or not set(cfg['methods']) <= set(METHODS):
        raise ValueError('Choose distinct methods from dtfd_afs, clam, promil')
    if not cfg['experiments'] or not set(cfg['experiments']) <= {'medical', 'mnist', 'achilles_sparsity'}:
        raise ValueError('Unknown or empty experiments')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('Use a simple run_name')
    if cfg.get('save_models') is not False:
        raise ValueError('This comparison exports predictions, not model weights')
    if cfg.get('epochs_override') is not None and (type(cfg['epochs_override']) is not int or cfg['epochs_override'] < 1):
        raise ValueError('epochs_override must be positive or null')
    if type(cfg['folds']) is not int or cfg['folds'] < 2:
        raise ValueError('folds must be an integer greater than one')
    if len(set(cfg['experiments'])) != len(cfg['experiments']):
        raise ValueError('Duplicate experiments')
    for key in ('cpus', 'feature_batch_size', 'mem_mb', 'runtime_minutes'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'Invalid {key}')
    for method in cfg['methods']:
        profile_list = list(cfg['profiles'][method].values()) if method == 'promil' else [cfg['profiles'][method]]
        for settings in profile_list:
            if type(settings['epochs']) is not int or settings['epochs'] < 1 or settings['lr'] <= 0 or settings['weight_decay'] < 0:
                raise ValueError(f'Invalid {method} training profile')
            if settings.get('patience') is not None and settings['patience'] < 1:
                raise ValueError('patience must be positive or null')
    if cfg['profiles']['dtfd_afs']['groups'] < 1 or cfg['profiles']['dtfd_afs']['inference_repeats'] < 1:
        raise ValueError('DTFD groups and inference_repeats must be positive')
    clam = cfg['profiles']['clam']
    if clam['k'] < 1 or not 0 < clam['bag_weight'] <= 1:
        raise ValueError('Invalid CLAM k or bag_weight')
    return cfg


def conditions(cfg):
    result = {}
    if 'medical' in cfg['experiments']:
        for dataset in cfg['medical_datasets']:
            if dataset not in ('achilles', 'ct', 'mri', 'xray'):
                raise ValueError(f'Unknown dataset: {dataset}')
            result['medical_' + dataset] = dict(dataset='medical_' + dataset, task=dataset)
    if 'mnist' in cfg['experiments']:
        for size, densities in cfg['mnist_densities_by_bag_size'].items():
            size = int(size)
            for density in densities:
                if size < 1 or not 0 < density <= 100 or size*density % 100:
                    raise ValueError('MNIST compositions must have whole target-image counts')
                key = f'mnist_n{size}_p{density}'
                if key in result:
                    raise ValueError(f'Duplicate condition: {key}')
                result[key] = dict(dataset='mnist', task='mnist', bag_size=size, density_percent=density)
    if 'achilles_sparsity' in cfg['experiments']:
        for size in cfg['achilles_bag_sizes']:
            if type(size) is not int or size < 3:
                raise ValueError('Invalid Achilles bag size')
            key = f'achilles_n{size}'
            if key in result:
                raise ValueError(f'Duplicate condition: {key}')
            result[key] = dict(dataset='achilles_sparsity', task='achilles', bag_size=size)
    return result


def source_files(cfg):
    """Inputs for Snakemake; missing saved designs are errors, never regenerated."""
    files = []
    project, results = Path(cfg['project_dir']), Path(cfg['results_dir'])
    for spec in conditions(cfg).values():
        if spec['dataset'].startswith('medical_'):
            import yaml
            path = project / 'config' / f"{spec['task']}_config.yaml"
            files.append(path)
            if path.is_file():
                data_cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
                base = path.parent / data_cfg['path_root'] if 'path_root' in data_cfg else project
                for field in ('metadata_csv', 'targets_csv'):
                    if data_cfg.get(field):
                        files.append((base / Path(data_cfg[field]).expanduser()).resolve())
        elif spec['dataset'] == 'achilles_sparsity':
            files.append(results / cfg['achilles_source_run'] / 'design.json')
        else:
            root = results / cfg['mnist_source_run']
            path = root / 'designs' / f"n{spec['bag_size']}" / f"p{spec['density_percent']}.json"
            files.append(path if path.exists() else root / 'design.json')
    return sorted({str(p) for p in files})


def check_partition(fold):
    """Check both source patients and donor/image identities across partitions."""
    image_sets, patients = {}, {}
    for split, bags in fold.items():
        if split not in ('train', 'val', 'test'):
            continue
        if not bags:
            raise ValueError(f'Empty {split} partition')
        patients[split] = {b['patient_id'] for b in bags}
        image_sets[split] = {json.dumps(i, sort_keys=True) for b in bags for i in b['images']}
    splits = list(image_sets)
    for j, left in enumerate(splits):
        for right in splits[j+1:]:
            if patients[left] & patients[right] or image_sets[left] & image_sets[right]:
                raise ValueError(f'Patient or image overlap between {left} and {right}')


def check_cv_coverage(folds):
    if any('test' in fold for fold in folds):
        return
    expected = {b['bag_id'] for split in folds[0].values() for b in split}
    held_out = []
    for fold in folds:
        present = {b['bag_id'] for split in fold.values() for b in split}
        if present != expected:
            raise ValueError('CV folds contain different source bag populations')
        held_out.extend(b['bag_id'] for b in fold['val'])
    if len(held_out) != len(expected) or set(held_out) != expected:
        raise ValueError('Each source bag must be held out exactly once across CV folds')


def prepare(cfg, root):
    import sys
    sys.path.insert(0, cfg['project_dir'])
    datasets, specs = {}, conditions(cfg)
    prepared = {}
    for name, spec in specs.items():
        dataset = spec['dataset']
        if dataset.startswith('medical_'):
            import yaml
            from configuration import load_config
            from datasets import build_manifest
            from utils import make_folds
            task = spec['task']
            # Override model selectors only; reuse all data filters and intensity settings.
            data_cfg = load_config(Path(cfg['project_dir']) / 'config' / f'{task}_config.yaml',
                                   overrides=dict(pretraining='imagenet', instance_load_clf=False, freeze_enc=True),
                                   net_overrides=dict(pooling_method='avg'))
            manifest = build_manifest(data_cfg)
            splits = make_folds(manifest, cfg['folds'], cfg['seed'])
            folds = []
            for train, val in splits:
                fold = {}
                for split, frame in [('train', train), ('val', val)]:
                    fold[split] = [dict(bag_id=str(bag), patient_id=str(group.patient_id.iloc[0]),
                                        label=int(group.label.iloc[0]), images=[dict(path=p) for p in group.path])
                                   for bag, group in frame.groupby('bag_id', sort=True)]
                folds.append(fold)
            datasets[dataset] = dict(task=task, preprocessing=data_cfg.get('preprocessing', {}),
                                     num_classes=3 if task == 'xray' else 2)
        elif dataset == 'achilles_sparsity':
            path = Path(cfg['results_dir']) / cfg['achilles_source_run'] / 'design.json'
            old = json.loads(path.read_text(encoding='utf-8'))
            folds = []
            for number in range(1, cfg['folds']+1):
                key = str(spec['bag_size'])
                if key not in old['designs'][str(number)]:
                    raise ValueError(f'Bag size {key} absent in {path}; select existing conditions')
                fold = {}
                for split, rows in old['designs'][str(number)][key].items():
                    owners = {r['patient_id'] for r in rows}
                    if any(i['patient_id'] not in owners for r in rows for i in r['images']):
                        raise ValueError('Saved Achilles donor belongs to another partition')
                    if any(len(r['images']) != spec['bag_size'] for r in rows):
                        raise ValueError('Saved Achilles bag has the wrong size')
                    fold[split] = [dict(bag_id=r['patient_id'], patient_id=r['patient_id'], label=r['label'],
                                        severity=r['severity'], source_count=r['source_count'],
                                        images=[dict(path=i['path']) for i in r['images']]) for r in rows]
                    if any(r['label'] != int(r['severity'] >= 2) for r in rows):
                        raise ValueError('Achilles source design must use classes 0/1 vs 2/3')
                folds.append(fold)
            datasets[dataset] = dict(task='achilles', preprocessing={}, num_classes=2)
        else:
            n, d = spec['bag_size'], spec['density_percent']
            source = Path(cfg['results_dir']) / cfg['mnist_source_run']
            path = source / 'designs' / f'n{n}' / f'p{d}.json'
            if not path.exists():
                path = source / 'design.json'
            old = json.loads(path.read_text(encoding='utf-8'))
            rows = old['conditions'][f'n{n}/p{d}']
            fold = {}
            for split in ('train', 'val', 'test'):
                fold[split] = [dict(bag_id=str(r['bag_id']), patient_id=f"{split}_{r['bag_id']}",
                                    label=r['label'], target_count=r['target_count'],
                                    images=[dict(mnist_split='test' if split == 'test' else 'train', index=i)
                                            for i in r['indices']]) for r in rows[split]]
                if any(len(r['indices']) != n or r['target_count'] != (n*d//100 if r['label'] else 0)
                       for r in rows[split]):
                    raise ValueError(f'Unexpected MNIST bag composition in {path}')
            folds = [fold]
            datasets[dataset] = dict(task='mnist', preprocessing={}, num_classes=2)
        for fold in folds:
            check_partition(fold)
            for rows in fold.values():
                if len({r['bag_id'] for r in rows}) != len(rows):
                    raise ValueError('Duplicate source bag IDs within a partition')
        check_cv_coverage(folds)
        prepared[name] = dict(spec, folds=folds)

    # Deduplicate image references across conditions. Frozen features are extracted once
    # per dataset and shared by CLAM/DTFD, without fitting anything on validation images.
    for dataset in datasets:
        unique = {json.dumps(i, sort_keys=True): i for condition in prepared.values() if condition['dataset'] == dataset
                  for fold in condition['folds'] for bags in fold.values() for bag in bags for i in bag['images']}
        keys = sorted(unique)
        datasets[dataset]['images'] = [unique[k] for k in keys]
        indices = {k: j for j, k in enumerate(keys)}
        for condition in prepared.values():
            if condition['dataset'] != dataset:
                continue
            for fold in condition['folds']:
                for bags in fold.values():
                    for bag in bags:
                        bag['images'] = [indices[json.dumps(i, sort_keys=True)] for i in bag['images']]
    for dataset in datasets.values():
        for image in dataset['images']:
            if 'path' in image and not Path(image['path']).is_file():
                raise FileNotFoundError(image['path'])
    source_hashes = {path: digest(path) for path in source_files(cfg)}
    write_json(root / 'manifest.json', dict(datasets=datasets, conditions=prepared, source_hashes=source_hashes))
    print(f'Prepared {len(prepared)} conditions with exact saved sparsity bags.', flush=True)
