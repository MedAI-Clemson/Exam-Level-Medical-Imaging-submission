"""Condition-specific inputs and verified reuse of the original completed runs."""
import csv
import hashlib
import json
from pathlib import Path


def combinations(cfg):
    if 'densities_by_bag_size' in cfg:
        if 'bag_sizes' in cfg or 'densities' in cfg:
            raise ValueError('Use densities_by_bag_size instead of bag_sizes/densities')
        grid = cfg['densities_by_bag_size']
        if not isinstance(grid, dict) or not grid:
            raise ValueError('densities_by_bag_size must be a nonempty mapping')
        result = []
        for size, densities in grid.items():
            # JSON object keys are strings, even when YAML used integer keys.
            if isinstance(size, str) and size.isdigit():
                size = int(size)
            if type(size) is not int or not isinstance(densities, list) or not densities:
                raise ValueError('Use integer bag sizes and nonempty density lists')
            result.extend((size, density) for density in densities)
    else:
        result = [(n, d) for n in cfg['bag_sizes'] for d in cfg['densities']]
    if len(result) != len(set(result)):
        raise ValueError('Duplicate conditions')
    for n, d in result:
        if type(n) is not int or type(d) is not int or n < 1 or not 0 < d <= 100 or n*d % 100:
            raise ValueError(f'Invalid whole-image composition: {n}, {d}')
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_stable(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True)
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def single_config(cfg, n, d):
    result = dict(cfg)
    for key in ('bag_sizes', 'densities', 'densities_by_bag_size', 'pooling'):
        result.pop(key, None)
    result.update(densities_by_bag_size={n: [d]}, pooling=['avg','max','attn','tanh_attn','transformer'])
    return result


def settings(cfg):
    # Only execution resources, destinations, and the sweep grid may vary during adoption.
    keys = ('target_digit','seed','train_bags','val_bags','test_bags','validation_fraction',
            'num_epochs','model_selection','save_models','encoder','num_heads','resolution',
            'drop_rate','learning_rate','weight_decay','scheduler_step','scheduler_gamma','batch_size')
    return dict({k: cfg[k] for k in keys}, freeze_encoder=cfg.get('freeze_encoder', False))


def code_hashes(cfg, base):
    project = Path(cfg['project_dir']).expanduser().resolve()
    files = [Path(base)/name for name in ('run_experiment.py','bags.py','sweep_support.py')]
    files += [project/name for name in ('models.py','pretraining.py')]
    return {p.name: digest(p) for p in files}


def migrate(cfg, root, base):
    """Validate old runs before creating external receipts. Never modify old results."""
    legacy = root/'design.json'
    if not legacy.exists():
        raise ValueError('Migration requires the original design.json')
    design = json.loads(legacy.read_text())
    legacy_hash = digest(legacy)
    pending = []
    for n, d in combinations(cfg):
        key = f'n{n}/p{d}'
        for pool in cfg['pooling']:
            if retained_result(cfg, root, base, n, d, pool):
                continue  # Never silently reapprove changed code or altered old results.
            marker = root/'completed'/key/f'{pool}.json'
            if not marker.exists():
                continue
            result = json.loads(marker.read_text())
            if result.get('design_sha256') != legacy_hash and (root/'designs'/f'n{n}'/f'p{d}.json').exists():
                continue  # A new-format result is already managed by normal dependencies.
            run = root/'runs'/key/pool
            old_cfg = json.loads((run/'config.resolved.json').read_text())
            if any(old_cfg.get(k) != v for k,v in dict(bag_size=n, density_percent=d, pooling_method=pool).items()):
                raise ValueError(f'{marker}: resolved condition mismatch')
            if settings(old_cfg) != settings(cfg):
                raise ValueError(f'{marker}: training settings differ; use a new run_name')
            expected = dict(bag_size=n, density_percent=d, pooling=pool, seed=cfg['seed'],
                            epochs=cfg['num_epochs'], pretraining='imagenet', model_selection='last',
                            encoder_state='frozen' if cfg.get('freeze_encoder',False) else 'trainable',
                            save_models=False, test_bags=cfg['test_bags'], design_sha256=legacy_hash)
            if any(result.get(k) != v for k,v in expected.items()) or old_cfg.get('design_sha256') != legacy_hash:
                raise ValueError(f'{marker}: metadata does not match original design/settings')
            condition = design['conditions'][key]
            predictions = run/'test_predictions.csv'
            with predictions.open(newline='') as f:
                rows = list(csv.DictReader(f))
            expected_bags = {r['bag_id']:r for r in condition['test']}
            if len(rows) != len(expected_bags) or {int(r['bag_id']) for r in rows} != set(expected_bags):
                raise ValueError(f'{predictions}: missing or duplicate bags')
            for r in rows:
                original = expected_bags[int(r['bag_id'])]
                if int(r['label']) != original['label'] or int(r['target_count']) != original['target_count'] or int(r['bag_size']) != n:
                    raise ValueError(f'{predictions}: bag metadata mismatch')
                if not 0 <= float(r['probability']) <= 1:
                    raise ValueError(f'{predictions}: invalid probability')
            receipt = dict(result=result, settings=settings(cfg), code=code_hashes(cfg,base),
                           files={str(p.resolve()):digest(p) for p in (legacy,marker,predictions,run/'config.resolved.json')})
            pending.append((root/'retained'/key/f'{pool}.json', receipt))
    # All candidates are checked before writing any receipts.
    for path, receipt in pending:
        write_stable(path, receipt)
    print(f'Verified and retained {len(pending)} completed runs. Original files are unchanged.')


def retained_result(cfg, root, base, n, d, pool):
    path = root/'retained'/f'n{n}'/f'p{d}'/f'{pool}.json'
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt['settings'] != settings(cfg) or receipt['code'] != code_hashes(cfg,base):
        raise ValueError(f'{path}: settings or model code changed; use a new run_name')
    if any(not Path(p).exists() or digest(p) != h for p,h in receipt['files'].items()):
        raise ValueError(f'{path}: original result/input changed; investigate before reuse')
    return path
