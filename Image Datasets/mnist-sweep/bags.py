"""Deterministic bag design. Source IDs, rather than image copies, are saved."""

import random
import re

METHODS = ('avg', 'max', 'attn', 'tanh_attn', 'transformer')


def validate(cfg):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('run_name must be a simple directory name')
    for key in ('train_bags', 'val_bags', 'test_bags'):
        if type(cfg[key]) is not int or cfg[key] < 2 or cfg[key] % 2:
            raise ValueError(f'{key} must be a positive even integer')
    for key in ('bag_sizes', 'densities', 'pooling'):
        if not cfg[key] or len(cfg[key]) != len(set(cfg[key])):
            raise ValueError(f'{key} must be nonempty and unique')
    if not set(cfg['pooling']) <= set(METHODS):
        raise ValueError('Unknown pooling method')
    for n in cfg['bag_sizes']:
        for density in cfg['densities']:
            if type(n) is not int or type(density) is not int or n < 1 or not 0 < density <= 100 or n * density % 100:
                raise ValueError('Every size/density combination must give an integer target count')
    if not 0 < cfg['validation_fraction'] < 1 or cfg['target_digit'] not in range(10):
        raise ValueError('Invalid validation fraction or target digit')
    if cfg['save_models'] is not False or cfg['model_selection'] != 'last':
        raise ValueError('This experiment uses last epoch and does not save models')
    for key in ('num_epochs', 'batch_size', 'resolution', 'num_heads', 'scheduler_step'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')


def source_split(labels, fraction, seed):
    # Split source images first, stratified by digit; no image crosses train/validation.
    rng = random.Random(seed)
    train, val = [], []
    for digit in range(10):
        ids = [i for i, label in enumerate(labels) if label == digit]
        rng.shuffle(ids)
        cut = int(len(ids) * fraction)
        if not 0 < cut < len(ids):
            raise ValueError('Each digit needs training and validation source images')
        val.extend(ids[:cut]); train.extend(ids[cut:])
    return train, val


def make_bags(labels, ids, count, size, density, target, seed):
    rng = random.Random(seed)
    by_digit = {d: [i for i in ids if labels[i] == d] for d in range(10)}
    if any(not pool for pool in by_digit.values()):
        raise ValueError('Every digit must be represented in each source split')
    distractors = [d for d in range(10) if d != target]
    rows = []
    for bag in range(count):
        label = bag % 2
        targets = size * density // 100 if label else 0
        instances = [rng.choice(by_digit[target]) for _ in range(targets)]
        instances += [rng.choice(by_digit[rng.choice(distractors)]) for _ in range(size - targets)]
        rng.shuffle(instances)
        rows.append(dict(bag_id=bag, label=label, target_count=targets, indices=instances))
    rng.shuffle(rows)
    return rows
