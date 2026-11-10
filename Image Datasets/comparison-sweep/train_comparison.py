"""Native method objectives and schedules, with matched bags and held-out evaluation."""

import copy
import hashlib
import json
import random
from pathlib import Path

from design import digest, write_csv, write_json


def metrics(rows, classes):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    labels = np.array([r['label'] for r in rows])
    probabilities = np.array([[r[f'probability_{c}'] for c in range(classes)] for r in rows])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid probabilities')
    predicted = (probabilities[:, 1] >= .5).astype(int) if classes == 2 else probabilities.argmax(1)
    cm = confusion_matrix(labels, predicted, labels=list(range(classes)))
    result = dict(n=len(rows), accuracy=float(np.trace(cm)/cm.sum()))
    recalls, specificities = [], []
    for c in range(classes):
        tp, fn = int(cm[c, c]), int(cm[c].sum()-cm[c, c])
        fp = int(cm[:, c].sum()-cm[c, c])
        tn = int(cm.sum()-tp-fn-fp)
        sensitivity = tp/(tp+fn) if tp+fn else None
        specificity = tn/(tn+fp) if tn+fp else None
        result.update({f'class_{c}_n': tp+fn, f'class_{c}_sensitivity': sensitivity,
                       f'class_{c}_specificity': specificity})
        if sensitivity is not None:
            recalls.append(sensitivity)
        if specificity is not None:
            specificities.append(specificity)
    result['balanced_accuracy'] = float(np.mean(recalls))
    result['sensitivity'] = result['class_1_sensitivity'] if classes == 2 else float(np.mean(recalls))
    result['specificity'] = result['class_1_specificity'] if classes == 2 else float(np.mean(specificities))
    result['auroc'] = None
    if len(set(labels)) == classes:
        result['auroc'] = float(roc_auc_score(labels, probabilities[:, 1]) if classes == 2 else
                                roc_auc_score(labels, probabilities, multi_class='ovr', average='macro'))
    result['brier'] = float(np.mean((probabilities[:, 1]-labels)**2) if classes == 2 else
                            np.mean(np.sum((probabilities-np.eye(classes)[labels])**2, axis=1)))
    if classes == 2:
        result.update(tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))
    for severity in range(4):
        selected = [r for r in rows if r.get('severity') == severity]
        if selected:
            correct = sum(int(r['probability_1'] >= .5) == r['label'] for r in selected)
            result[f'severity_{severity}_n'] = len(selected)
            result[f'severity_{severity}_correct_rate'] = correct/len(selected)
    return result


def profile(cfg, method, mnist):
    settings = copy.deepcopy(cfg['profiles'][method])
    if method == 'promil':
        settings = settings['mnist' if mnist else 'medical']
    if cfg.get('epochs_override') is not None:
        settings['epochs'] = cfg['epochs_override']
    return settings


def training_order(size, count=None):
    """Visit each bag once in shuffled order, optionally truncating for a smoke test."""
    count = size if count is None else count
    if size < 1 or type(count) is not int or count < 1:
        raise ValueError('Training size and update budget must be positive')
    order = list(range(size))
    random.shuffle(order)
    return order[:min(size, count)]


def train(cfg, root, condition_name, method, fold_number=None, training_bags_per_epoch=None):
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet18, ResNet18_Weights
    from comparator_models import CLAM, DTFDAFS, LeNet, ProMIL
    from image_data import ImageReader, ProMILBags
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError('Submit training through Slurm on a GPU')
    torch.set_num_threads(cfg['cpus'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    condition = manifest['conditions'][condition_name]
    dataset = manifest['datasets'][condition['dataset']]
    classes, mnist = dataset['num_classes'], dataset['task'] == 'mnist'
    settings = profile(cfg, method, mnist)
    directory = root / 'runs' / condition_name / method
    marker = root / 'completed' / condition_name / f'{method}.json'
    if fold_number is not None:
        if type(fold_number) is not int or not 1 <= fold_number <= len(condition['folds']):
            raise ValueError('Requested fold is outside the saved design')
        directory = directory / f'fold{fold_number}'
        marker = root / 'completed' / condition_name / method / f'fold{fold_number}.json'
    requested_fold = fold_number
    marker.unlink(missing_ok=True)
    features = None
    if method != 'promil':
        cache = root / 'features' / condition['dataset']
        receipt = json.loads((cache / 'complete.json').read_text(encoding='utf-8'))
        if receipt['manifest_sha256'] != digest(manifest_path):
            raise ValueError('Feature cache does not match the current image manifest')
        features = np.load(cache / 'features.npy', mmap_mode='r')
    reader = ImageReader(cfg, dataset) if method == 'promil' else None
    write_json(directory / 'config.resolved.json', dict(sweep=cfg, method=method, profile=settings,
                                                       condition=condition_name, manifest_sha256=digest(manifest_path)))
    all_predictions, fold_results = [], []

    for fold_number, splits in enumerate(condition['folds'], start=1):
        if requested_fold is not None and fold_number != requested_fold:
            continue
        seed = cfg['seed'] + fold_number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if method == 'dtfd_afs':
            model = DTFDAFS(classes, groups=settings['groups']).cuda()
            optimizers = [torch.optim.Adam(parameters, lr=settings['lr'], weight_decay=settings['weight_decay'])
                          for parameters in (model.tier1_parameters(), model.tier2_parameters())]
            schedulers = [torch.optim.lr_scheduler.MultiStepLR(o, settings['milestones'], gamma=settings['gamma']) for o in optimizers]
        elif method == 'clam':
            model = CLAM(classes, k=settings['k'], dropout=settings['dropout'], bag_weight=settings['bag_weight']).cuda()
            optimizers = [torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])]
            schedulers = []
        else:
            if mnist:
                encoder, width = LeNet(), 500
            else:
                encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                encoder.fc, width = nn.Identity(), 512
            model = ProMIL(encoder, width, classes, mnist, settings['dropout']).cuda()
            optimizers = [torch.optim.Adam(model.parameters(), lr=settings['lr'], betas=tuple(settings['betas']),
                                            weight_decay=settings['weight_decay'])]
            schedulers = []

        # Workers decode/transform upcoming bags while the GPU processes the current bag.
        # Separate split datasets preserve the existing training/validation transforms.
        loaders, load_orders = {}, {}
        if method == 'promil':
            for part, bags in splits.items():
                load_orders[part] = list(range(len(bags)))
                loaders[part] = DataLoader(
                    ProMILBags(reader, bags, part == 'train'), batch_size=None,
                    sampler=load_orders[part], num_workers=cfg['num_workers'], pin_memory=True,
                    persistent_workers=cfg['num_workers'] > 0,
                    generator=torch.Generator().manual_seed(seed + (part != 'train')))

        def epoch(split, updating):
            model.train(updating)
            bags = splits[split]
            order = list(range(len(bags)))
            if updating:
                order = training_order(len(bags), training_bags_per_epoch)
            rows, losses = [], []
            batches = ((index, None) for index in order)
            if method == 'promil':
                load_orders[split][:] = order
                batches = loaders[split]
            for index, loaded_images in batches:
                bag = bags[index]
                label = torch.tensor(bag['label'], device='cuda', dtype=torch.long)
                if features is not None:
                    inputs = torch.from_numpy(np.array(features[bag['images']], copy=True)).cuda()
                else:
                    inputs = loaded_images.cuda(non_blocking=True)
                with torch.set_grad_enabled(updating):
                    if method == 'dtfd_afs':
                        if updating:
                            loss, probability, detail = model.train_step(inputs, label, optimizers, settings['clip'])
                        else:
                            key = f"{seed}|{bag['patient_id']}|{bag['bag_id']}".encode()
                            bag_seed = int.from_bytes(hashlib.sha256(key).digest()[:4], 'big')
                            probability = model.predict(inputs, bag_seed, settings['inference_repeats'])
                            loss = -probability[0, label].clamp_min(torch.finfo(probability.dtype).tiny).log()
                    else:
                        if updating:
                            optimizers[0].zero_grad(set_to_none=True)
                        if method == 'clam' and not updating:
                            logits, _, _ = model(inputs)
                            probability = logits.softmax(-1)
                            loss = F.cross_entropy(logits, label.reshape(1))
                        else:
                            loss, probability, detail = model.objective(inputs, label)
                        if not torch.isfinite(loss):
                            raise ValueError(f'Non-finite {method} loss')
                        if updating:
                            loss.backward()
                            optimizers[0].step()
                if not torch.isfinite(loss) or not torch.isfinite(probability).all():
                    raise ValueError('Non-finite loss or prediction')
                values = probability.detach().cpu().reshape(-1).tolist()
                row = {k: v for k, v in bag.items() if k != 'images'}
                row.update(fold=fold_number, method=method, condition=condition_name, num_images=len(bag['images']))
                row.update({f'probability_{c}': p for c, p in enumerate(values)})
                if classes == 2:
                    row['probability'] = values[1]
                rows.append(row)
                losses.append(float(loss.detach().cpu()))
            return dict(loss=float(np.mean(losses)), **metrics(rows, classes)), rows

        history, best_state, best_loss, best_epoch, stale = [], None, float('inf'), 0, 0
        # Only MNIST has a separate validation set for early stopping/checkpoint choice.
        # Medical CV evaluates the fixed final epoch to avoid selecting on held-out folds.
        for number in range(1, settings['epochs']+1):
            if method == 'promil':
                model.set_encoder_frozen(not mnist and number <= settings['warmup_epochs'])
            training, _ = epoch('train', True)
            validation, validation_rows = epoch('val', False)
            history.append(dict(epoch=number, **{f'train_{k}': v for k, v in training.items()},
                                **{f'val_{k}': v for k, v in validation.items()}))
            write_csv(directory / f'fold{fold_number}' / 'history.csv', history)
            print(f'{condition_name} {method} fold {fold_number} epoch {number}: '
                  f'train loss {training["loss"]:.4f}; val loss {validation["loss"]:.4f}', flush=True)
            for scheduler in schedulers:
                scheduler.step()
            if mnist and settings.get('patience'):
                if validation['loss'] < best_loss:
                    best_loss, best_epoch, stale = validation['loss'], number, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                if stale >= settings['patience'] and number >= settings.get('minimum_epochs', 0):
                    break

        selected_epoch = number
        if mnist:
            if best_state is not None:
                model.load_state_dict(best_state)
                selected_epoch = best_epoch
            final_metrics, final_rows = epoch('test', False)
        else:
            final_metrics, final_rows = validation, validation_rows
        if method == 'promil':
            for channel, alpha in enumerate(model.estimator.detach().sigmoid().cpu().tolist()):
                final_metrics[f'bernstein_alpha_{channel}'] = alpha
        for row in final_rows:
            row['selected_epoch'] = selected_epoch
        fold_results.append(dict(fold=fold_number, selected_epoch=selected_epoch,
                                 training_bags=len(splits['train']),
                                 bags_per_reference_epoch=training_bags_per_epoch or len(splits['train']),
                                 optimizer_updates=number*(training_bags_per_epoch or len(splits['train']))*(2 if method == 'dtfd_afs' else 1),
                                 **final_metrics))
        all_predictions.extend(final_rows)
        write_csv(directory / f'fold{fold_number}' / 'predictions.csv', final_rows)
        del model, optimizers, schedulers, best_state, loaders
        torch.cuda.empty_cache()

    write_csv(directory / 'fold_metrics.csv', fold_results)
    write_csv(directory / 'predictions.csv', all_predictions)
    result = dict(condition=condition_name, method=method, dataset=condition['dataset'], num_classes=classes,
                  pooling=method, epoch_budget=settings['epochs'], threshold=.5 if classes == 2 else None,
                  encoder_profile=('lenet_trainable' if mnist else 'resnet18_imagenet_two_stage') if method == 'promil'
                                  else 'resnet50_imagenet_layer3_frozen',
                  model_selection='best_val_loss' if mnist and settings.get('patience') else 'last', save_models=False,
                  folds=len(fold_results), evaluation='test' if mnist else 'out_of_fold',
                  run_dir=str(directory), manifest_sha256=digest(manifest_path),
                  **{k: v for k, v in condition.items() if k in ('bag_size', 'density_percent')},
                  **{f'pooled_{k}': v for k, v in metrics(all_predictions, classes).items()})
    from scipy.stats import t
    for metric in ('auroc', 'sensitivity', 'specificity', 'balanced_accuracy', 'brier'):
        values = [f[metric] for f in fold_results if f[metric] is not None]
        mean = float(np.mean(values)) if values else None
        margin = float(t.ppf(.95, len(values)-1)*np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        result.update({f'{metric}_mean': mean, f'{metric}_ci90_low': mean-margin if margin is not None else None,
                       f'{metric}_ci90_high': mean+margin if margin is not None else None})
    write_json(directory / 'summary.json', result)
    write_json(marker, result)
