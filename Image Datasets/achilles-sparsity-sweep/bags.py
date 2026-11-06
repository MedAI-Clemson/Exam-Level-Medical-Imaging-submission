"""Read expert selections and construct reproducible, patient-separated bags."""

import csv
import hashlib
import json
import random
import re
from pathlib import Path


METHODS = ('avg', 'max', 'attn', 'tanh_attn', 'transformer')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return
    temporary = path.with_suffix('.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(cfg):
    if cfg['folds'] != 10:
        raise ValueError('This experiment uses 10 patient folds')
    sizes = cfg['bag_sizes']
    if not sizes or len(sizes) != len(set(sizes)) or any(type(n) is not int or n < 3 or n > 50 for n in sizes):
        raise ValueError('Use distinct bag sizes between 3 and 50')
    if not cfg['pooling'] or not set(cfg['pooling']) <= set(METHODS) or len(cfg['pooling']) != len(set(cfg['pooling'])):
        raise ValueError('Invalid or duplicate pooling methods')
    if cfg['model_selection'] != 'last' or cfg['save_models'] is not False:
        raise ValueError('Use last-epoch evaluation and save_models: false')
    if type(cfg['freeze_encoder']) is not bool:
        raise ValueError('freeze_encoder must be true or false')
    if type(cfg['num_epochs']) is not int or cfg['num_epochs'] < 1:
        raise ValueError('num_epochs must be a positive integer')
    if not 0 < cfg['threshold'] < 1:
        raise ValueError('threshold must lie between zero and one')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_name']):
        raise ValueError('Use a simple run_name')
    for key in ('batch_size', 'resolution', 'cpus', 'num_heads', 'scheduler_step'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for method, heads in cfg['num_heads_by_pooling'].items():
        if method not in METHODS or type(heads) is not int or heads < 1:
            raise ValueError('Invalid per-method head count')
    if cfg['num_workers'] < 0 or cfg['learning_rate'] <= 0 or cfg['weight_decay'] < 0:
        raise ValueError('Invalid loader or optimizer settings')


def design_settings(cfg):
    return {k: cfg[k] for k in ('data_dir', 'metadata_csv', 'image_number_regex', 'image_extensions',
                               'seed', 'folds', 'bag_sizes')}


def read_patients(path):
    """Zero is a valid image number; only blank annotation cells are missing."""
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        id_column = next((c for c in columns if c.strip() in ('PatiendID', 'PatientID', 'patient_id')), None)
        label_column = next((c for c in columns if c.strip().startswith('Label:')), None)
        rank_columns = [next((c for c in columns if c.strip().startswith(prefix)), None)
                        for prefix in ('1st Most Useful', '2nd Most Useful', '3rd Most Useful')]
        if not id_column or not label_column or not all(rank_columns):
            raise ValueError('Missing patient, severity, or ranked-image columns')
        patients, excluded, seen = [], [], set()
        for row in reader:
            patient = row[id_column].strip()
            if not patient or patient in seen:
                raise ValueError(f'Missing or duplicate patient ID: {patient}')
            seen.add(patient)
            severity = int(row[label_column].strip())
            if severity not in range(4):
                raise ValueError(f'{patient}: severity must be 0, 1, 2, or 3')
            selected = [int(row[c].strip()) for c in rank_columns if row[c].strip()]
            if len(selected) != len(set(selected)) or any(i < 0 for i in selected):
                raise ValueError(f'{patient}: invalid or duplicate selected image numbers')
            if not selected:
                excluded.append(dict(patient_id=patient, severity=severity, reason='no selected images'))
                continue
            patients.append(dict(patient_id=patient, severity=severity, label=int(severity >= 2), selected=selected))
    return patients, excluded


def image_inventory(cfg, patients):
    """Map explicit filename numbers, never an implicit filesystem listing order."""
    root = Path(cfg['data_dir']).expanduser().resolve()
    pattern = re.compile(cfg['image_number_regex'])
    inventory, ownership = {}, {}
    for patient in patients:
        patient_id = patient['patient_id']
        directory = (root / patient_id).resolve()
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError(f'Missing or invalid patient directory: {directory}')
        images = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in cfg['image_extensions']:
                continue
            match = pattern.fullmatch(path.stem)
            if not match:
                raise ValueError(f'Cannot identify image number from {path.name}; set image_number_regex')
            number = int(match.group('image_number'))
            resolved = str(path.resolve())
            if number in images or resolved in ownership:
                raise ValueError(f'Duplicate image number or shared patient image: {path}')
            images[number] = resolved
            ownership[resolved] = patient_id
        if not images:
            raise ValueError(f'No eligible images: {directory}')
        missing = set(patient['selected']) - set(images)
        if missing:
            raise ValueError(f'{patient_id}: selected image numbers not found: {sorted(missing)}')
        inventory[patient_id] = images
    return inventory


def assign_folds(patients, folds, seed):
    """Stratify the binary label; each patient is held out exactly once."""
    rng, assignments, offset = random.Random(seed), {}, 0
    for label in (0, 1):
        ids = sorted(p['patient_id'] for p in patients if p['label'] == label)
        if len(ids) < folds:
            raise ValueError(f'Need at least {folds} eligible patients in binary class {label}')
        rng.shuffle(ids)
        for i, patient in enumerate(ids):
            assignments[patient] = (offset + i) % folds + 1
        offset = (offset + len(ids)) % folds
    return assignments


def seeded_rng(seed, *parts):
    key = '|'.join(map(str, (seed, *parts))).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'big'))


def construct(patients, inventory, assignments, cfg):
    """One fixed donor sequence per source/fold; smaller bags use its prefix."""
    sizes = sorted(cfg['bag_sizes'])
    designs, shortages = {}, []
    for fold in range(1, cfg['folds'] + 1):
        designs[str(fold)] = {str(n): {} for n in sizes}
        for split in ('train', 'val'):
            members = [p for p in patients if (assignments[p['patient_id']] == fold) == (split == 'val')]
            pool = [dict(patient_id=p['patient_id'], image_number=i, path=path)
                    for p in members if p['label'] == 0
                    for i, path in sorted(inventory[p['patient_id']].items())]
            rows_by_size = {n: [] for n in sizes}
            for patient in members:
                patient_id = patient['patient_id']
                source = [dict(patient_id=patient_id, image_number=i, path=inventory[patient_id][i], role='source')
                          for i in patient['selected']]
                donors = [dict(image, role='distractor') for image in pool if image['patient_id'] != patient_id]
                needed = max(sizes) - len(source)
                if len(donors) < needed:
                    shortages.append(dict(fold=fold, split=split, patient_id=patient_id,
                                          available=len(donors), required=needed,
                                          largest_feasible_bag=len(donors) + len(source)))
                    continue
                seeded_rng(cfg['seed'], fold, split, patient_id, 'donors').shuffle(donors)
                for size in sizes:
                    images = source + donors[:size - len(source)]
                    seeded_rng(cfg['seed'], fold, split, patient_id, size, 'order').shuffle(images)
                    rows_by_size[size].append(dict(patient_id=patient_id, severity=patient['severity'],
                                                   label=patient['label'], source_count=len(source), images=images))
            for size in sizes:
                designs[str(fold)][str(size)][split] = rows_by_size[size]
    return designs, shortages


def prepare(cfg, root):
    patients, excluded = read_patients(cfg['metadata_csv'])
    inventory = image_inventory(cfg, patients)
    assignments = assign_folds(patients, cfg['folds'], cfg['seed'])
    designs, shortages = construct(patients, inventory, assignments, cfg)
    report = dict(eligible_patients=len(patients), healthy=sum(p['label'] == 0 for p in patients),
                  unhealthy=sum(p['label'] == 1 for p in patients), excluded=excluded,
                  eligible_images=sum(map(len, inventory.values())), shortages=shortages)
    write_json(root / 'preparation_report.json', report)
    if shortages:
        smallest = min(r['largest_feasible_bag'] for r in shortages)
        raise ValueError(f'Insufficient distinct healthy donors. See preparation_report.json; '
                         f'the smallest affected bag capacity is {smallest}. No duplication or cross-partition borrowing was used.')
    # Record the actual file mapping and sizes for audit, without copying source images.
    files = [dict(patient_id=p, image_number=i, path=f, bytes=Path(f).stat().st_size)
             for p, images in inventory.items() for i, f in sorted(images.items())]
    write_json(root / 'design.json', dict(configuration=design_settings(cfg), metadata_sha256=digest(cfg['metadata_csv']),
                                        folds=assignments, patients=patients, inventory=files, designs=designs))
    print(json.dumps(report, indent=2), flush=True)
