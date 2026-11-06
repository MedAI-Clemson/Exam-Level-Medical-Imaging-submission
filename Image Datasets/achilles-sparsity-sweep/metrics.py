"""Binary detection metrics with explicit denominators and fold summaries."""

import math
import statistics


def evaluate(rows, threshold):
    positives = [r for r in rows if r['label'] == 1]
    negatives = [r for r in rows if r['label'] == 0]
    if not positives or not negatives:
        raise ValueError('Both binary classes are required for evaluation')
    if any(not math.isfinite(r['probability']) or not 0 <= r['probability'] <= 1 for r in rows):
        raise ValueError('Non-finite or invalid prediction probability')
    tp = sum(r['probability'] >= threshold for r in positives)
    tn = sum(r['probability'] < threshold for r in negatives)
    # Pairwise concordance equals AUROC, including half credit for ties.
    auc = sum((p['probability'] > n['probability']) + .5 * (p['probability'] == n['probability'])
              for p in positives for n in negatives) / (len(positives) * len(negatives))
    result = dict(n=len(rows), n_healthy=len(negatives), n_unhealthy=len(positives),
                  tp=tp, fn=len(positives)-tp, tn=tn, fp=len(negatives)-tn,
                  sensitivity=tp / len(positives), specificity=tn / len(negatives),
                  balanced_accuracy=(tp / len(positives) + tn / len(negatives)) / 2,
                  auroc=auc, brier=statistics.mean((r['probability'] - r['label']) ** 2 for r in rows))
    for severity in range(4):
        subgroup = [r for r in rows if r['severity'] == severity]
        result[f'class_{severity}_n'] = len(subgroup)
        name = 'specificity' if severity < 2 else 'sensitivity'
        correct = sum((r['probability'] >= threshold) == bool(r['label']) for r in subgroup)
        result[f'class_{severity}_{name}'] = correct / len(subgroup) if subgroup else None
    return result


def fold_summary(rows):
    """Two-sided 90% t intervals across ten folds; not independent test-set CIs."""
    if len(rows) != 10 or {r['fold'] for r in rows} != set(range(1, 11)):
        raise ValueError('Expected ten distinct completed folds')
    result = {}
    for metric in ('auroc', 'sensitivity', 'specificity', 'balanced_accuracy', 'brier'):
        values = [r[metric] for r in rows]
        mean = statistics.mean(values)
        margin = 1.8331129326536335 * statistics.stdev(values) / math.sqrt(10)
        result.update({f'{metric}_mean': mean, f'{metric}_ci90_low': mean-margin,
                       f'{metric}_ci90_high': mean+margin})
    return result
