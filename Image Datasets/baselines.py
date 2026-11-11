"""Instance-label training with mean-probability evaluation, and noisy-OR MIL."""

import torch
from torch.nn import functional as F


def baseline_batch(model, images, lengths, labels, method, training):
    """Return loss, bag probabilities, and the loss's number of observations.

    Instance training uses every image occurrence with its source bag label.
    Validation instead scores the mean probability against the true bag label.
    SA uses the existing stable noisy-OR implementation in models.py.
    """
    if method not in ('instance', 'stan'):
        raise ValueError(f'Unknown baseline: {method}')
    logits = model(images, lengths)
    binary = model.num_out == 1
    probabilities = logits.sigmoid() if binary else logits.softmax(-1)
    targets = labels
    if method == 'instance':
        probabilities = torch.stack([p.mean(0) for p in probabilities.split(lengths)])
        if training:
            targets = labels.repeat_interleave(torch.as_tensor(lengths, device=labels.device))
        else:
            loss = (F.binary_cross_entropy(probabilities[:, 0], labels.float()) if binary
                    else F.nll_loss(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(), labels.long()))
            return loss, probabilities, len(labels)
    loss = (F.binary_cross_entropy_with_logits(logits[:, 0], targets.float()) if binary
            else F.cross_entropy(logits, targets.long()))
    return loss, probabilities, len(targets)


def run_baseline_epoch(model, loader, device, optimizer=None, *, method):
    import numpy as np
    import pandas as pd
    from utils import classification_metrics

    training = optimizer is not None
    model.train(training)
    labels_list, probabilities_list, records = [], [], []
    total_loss, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            labels = batch['labels'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, probabilities, weight = baseline_batch(model, batch['images'].to(device),
                                                         batch['lengths'], labels, method, training)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite baseline loss')
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * weight
            count += weight
            labels_list.append(labels.detach().cpu().numpy())
            probabilities_list.append(probabilities.detach().cpu().numpy())
            records.extend(zip(batch['patient_id'], batch['bag_id'], batch['instance_path']))
    if not count:
        raise ValueError('Empty baseline loader')
    labels, probabilities = np.concatenate(labels_list), np.concatenate(probabilities_list)
    predictions = pd.DataFrame(records, columns=['patient_id', 'bag_id', 'instance_path'])
    predictions['label'] = labels
    for column in range(model.num_out):
        predictions[f'probability_{1 if model.num_out == 1 else column}'] = probabilities[:, column]
    return dict(loss=total_loss/count, **classification_metrics(labels, probabilities)), predictions
