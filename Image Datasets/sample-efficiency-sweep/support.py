"""Patient subsets and configuration checks; no training-library imports."""

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path

POOLING = ('avg', 'max', 'attn', 'tanh_attn', 'transformer')
COMPARATORS = ('dtfd_afs', 'clam', 'promil')
DATASETS = ('achilles', 'ct', 'mri', 'xray')


def resolve_config(raw, base):
    cfg = copy.deepcopy(raw)
    for key in ('project_dir', 'results_dir'):
        cfg[key] = str((Path(base) / Path(cfg[key]).expanduser()).resolve())
    for key, allowed in [('methods', POOLING + COMPARATORS), ('medical_datasets', DATASETS)]:
        values = cfg[key]
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f'Invalid or duplicate {key}')
    if cfg['folds'] != 10 or type(cfg['folds']) is not int:
        raise ValueError('This experiment requires 10 folds')
    for key in ('seed',):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('sampling_seeds', 'train_percentages'):
        values = cfg[key]
        if not values or len(set(values)) != len(values) or any(type(v) is not int for v in values):
            raise ValueError(f'Invalid or duplicate {key}')
    if any(v < 0 for v in cfg['sampling_seeds']):
        raise ValueError('Sampling seeds must be nonnegative')
    if any(not 0 < v <= 100 for v in cfg['train_percentages']):
        raise ValueError('Training percentages must be in (0, 100]')
    if cfg.get('save_models') is not False:
        raise ValueError('Model saving is disabled for this workflow')
    for key in ('cpus', 'feature_batch_size', 'mem_mb', 'runtime_minutes'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['num_workers']) is not int or cfg['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    for key in ('epochs_override', 'smoke_bags_per_epoch'):
        if cfg.get(key) is not None and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f'{key} must be positive or null')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('Use a simple run_name')
    for method in set(cfg['methods']) & set(COMPARATORS):
        profile = cfg['profiles'][method]
        if method == 'promil':
            profile = profile['medical']
        if type(profile['epochs']) is not int or profile['epochs'] < 1 or profile['lr'] <= 0 or profile['weight_decay'] < 0:
            raise ValueError(f'Invalid {method} training budget or optimizer')
        if method == 'promil' and (type(profile['warmup_epochs']) is not int or not 0 <= profile['warmup_epochs'] <= profile['epochs']):
            raise ValueError('Invalid ProMIL warmup budget')
    return cfg


def conditions(cfg):
    return {f'{d}_p{p}_s{s}': dict(dataset='medical_' + d, task=d, train_percent=p, sampling_seed=s)
            for d in cfg['medical_datasets'] for p in cfg['train_percentages'] for s in cfg['sampling_seeds']}


def immutable_json(path, value):
    """A run name identifies one design; never silently replace its provenance."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'{path} differs from the saved run. Use a new run_name.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def nested_subsets(bags, percentages, seed):
    """Stratify by patient label signature, retaining every bag of selected patients.

    A signature handles patients with bags from multiple classes without splitting
    them. Ceiling within each stratum guarantees class coverage at small fractions.
    The actual patient count can therefore slightly exceed the nominal percentage.
    """
    labels = {}
    for bag in bags:
        labels.setdefault(bag['patient_id'], set()).add(bag['label'])
    if not labels:
        raise ValueError('Empty training partition')
    strata = {}
    for patient, signature in labels.items():
        strata.setdefault(tuple(sorted(signature)), []).append(patient)
    ordered = {}
    for signature, patients in sorted(strata.items()):
        patients = sorted(patients)
        token = f'{seed}|{signature}'.encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], 'big'))
        rng.shuffle(patients)
        ordered[signature] = patients
    result = {}
    for percent in percentages:
        if type(percent) is not int or not 0 < percent <= 100:
            raise ValueError('Invalid training percentage')
        selected = {p for patients in ordered.values() for p in patients[:math.ceil(len(patients)*percent/100)]}
        result[percent] = [copy.deepcopy(b) for b in bags if b['patient_id'] in selected]
    return result


class SinglePassBatchSampler:
    """One shuffled pass per epoch, optionally truncated for smoke tests."""
    def __init__(self, size, batch_size, bag_budget, seed):
        if min(size, batch_size, bag_budget) < 1:
            raise ValueError('Sampler sizes must be positive')
        self.size, self.batch_size, self.bag_budget = size, batch_size, min(size, bag_budget)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return math.ceil(self.bag_budget / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(self.size))
        rng.shuffle(indices)
        indices = indices[:self.bag_budget]
        for start in range(0, len(indices), self.batch_size):
            yield indices[start:start+self.batch_size]


def prepare(cfg, root):
    from design import prepare as prepare_full, check_partition, digest, write_csv
    # Reuse the same medical manifest construction and grouped CV as existing runs.
    full_cfg = dict(cfg, experiments=['medical'])
    full_root = root / '_full_design'
    prepare_full(full_cfg, full_root)
    full = json.loads((full_root / 'manifest.json').read_text(encoding='utf-8'))
    prepared, report = {}, []
    for name, spec in conditions(cfg).items():
        source = full['conditions'][spec['dataset']]
        folds, budgets = [], []
        for number, partition in enumerate(source['folds'], 1):
            seed = int.from_bytes(hashlib.sha256(f"{spec['task']}|{number}|{spec['sampling_seed']}".encode()).digest()[:8], 'big')
            selected = nested_subsets(partition['train'], [spec['train_percent']], seed)[spec['train_percent']]
            fold = dict(train=selected, val=copy.deepcopy(partition['val']))
            check_partition(fold)
            classes = set(b['label'] for b in partition['train'])
            if set(b['label'] for b in selected) != classes or set(b['label'] for b in fold['val']) != classes:
                raise ValueError('A partition is missing a class')
            full_patients = len({b['patient_id'] for b in partition['train']})
            selected_patients = len({b['patient_id'] for b in selected})
            info = dict(condition=name, dataset=spec['task'], fold=number,
                        train_percent=spec['train_percent'], sampling_seed=spec['sampling_seed'],
                        full_train_patients=full_patients, train_patients=selected_patients,
                        actual_train_percent=100*selected_patients/full_patients,
                        full_train_bags=len(partition['train']), train_bags=len(selected),
                        val_patients=len({b['patient_id'] for b in fold['val']}), val_bags=len(fold['val']))
            for label in sorted(classes):
                info[f'train_class_{label}_patients'] = len({b['patient_id'] for b in selected if b['label']==label})
            report.append(info)
            folds.append(fold)
            budgets.append(len(partition['train']))
        prepared[name] = dict(spec, folds=folds, full_train_bags=budgets)
    # Snapshot the pooling configs, overriding pretraining/freezing before validation.
    from configuration import load_config
    pool_configs = {}
    for task in cfg['medical_datasets']:
        pool_configs[task] = load_config(Path(cfg['project_dir']) / 'config' / f'{task}_config.yaml',
            overrides=dict(pretraining='imagenet', instance_load_clf=False, freeze_enc=True,
                           freeze_clf=False, mil=True, save_models=False, model_selection='last',
                           seed=cfg['seed'], folds=10), net_overrides=dict(pooling_method='avg'))
    manifest = dict(datasets=full['datasets'], conditions=prepared, source_hashes=full['source_hashes'],
                    pooling_configs=pool_configs, split_seed=cfg['seed'])
    immutable_json(root / 'manifest.json', manifest)
    write_csv(root / 'subset_counts.csv', report)
    immutable_json(root / 'preparation_report.json', dict(conditions=len(prepared), folds=10,
                   subsets=len(report), all_classes_present=True, patient_and_image_partitions_disjoint=True,
                   manifest_sha256=digest(root / 'manifest.json')))
    print(f'Prepared {len(report)} nested training subsets; see subset_counts.csv for actual counts.', flush=True)
