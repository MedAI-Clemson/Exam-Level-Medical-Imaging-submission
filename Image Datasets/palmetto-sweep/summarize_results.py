"""Summarize completed MIL jobs, keeping encoder states separate (also reads older sweeps)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
import yaml


def collect(root):
    rows, seen = [], set()
    for marker in sorted((root / "completed").rglob("*.json")):
        record = json.loads(marker.read_text())
        if record.get("stage") != "mil":
            continue
        output = Path(record["output_dir"])
        with (output / "config.yaml").open() as stream:
            cfg = yaml.safe_load(stream)
        state = "frozen" if cfg["freeze_enc"] else "trainable"
        if record.get("encoder_state", state) != state:
            raise ValueError(f"Encoder state disagrees with resolved config: {marker}")
        key = (record["dataset"], record["pretraining"], record["pooling"], state)
        if key in seen:
            raise ValueError(f"Duplicate completed configuration: {key}")
        seen.add(key)

        summary = pd.read_csv(output / "summary.csv")
        predictions = pd.read_csv(output / "predictions.csv")
        expected = set(range(1, record["folds"] + 1))
        if set(summary.fold) != expected or len(summary) != len(expected) or set(predictions.fold) != expected:
            raise ValueError(f"Incomplete or duplicate folds: {marker}")
        probability_columns = sorted(
            (c for c in predictions if c.startswith("probability_")),
            key=lambda c: int(c.rsplit("_", 1)[1]))
        for fold in summary.itertuples(index=False):
            part = predictions.loc[predictions.fold == fold.fold]
            probabilities = part[probability_columns].to_numpy()
            labels = part.label.to_numpy(dtype=int)
            if probabilities.shape[1] == 1:
                brier = np.mean((probabilities[:, 0] - labels) ** 2)
            else:
                # Multiclass Brier is the sum of squared class errors (range 0–2).
                truth = np.eye(probabilities.shape[1])[labels]
                brier = np.mean(np.sum((probabilities - truth) ** 2, axis=1))
            if not np.isfinite([fold.roc_auc, brier]).all():
                raise ValueError(f"Non-finite metric: {marker}, fold {fold.fold}")
            rows.append(dict(dataset=key[0], pretraining=key[1], pooling=key[2],
                             encoder_state=state, fold=fold.fold, auroc=fold.roc_auc,
                             brier=brier, output_dir=str(output)))
    if not rows:
        raise ValueError(f"No completed MIL jobs under {root / 'completed'}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Sweep folder containing completed/ and attempts/")
    args = parser.parse_args()
    root = args.results.expanduser().resolve()
    scores = collect(root)
    scores.to_csv(root / "metrics_per_fold.csv", index=False)
    for metric in ("auroc", "brier"):
        rows = []
        for key, group in scores.groupby(["dataset", "pretraining", "pooling", "encoder_state"]):
            values = group[metric]
            n, mean = len(values), values.mean()
            margin = t.ppf(0.95, n - 1) * values.std(ddof=1) / np.sqrt(n)
            rows.append(dict(zip(["dataset", "pretraining", "pooling", "encoder_state"], key),
                             **{f"mean_{metric}": mean, "ci90_lower": mean - margin,
                                "ci90_upper": mean + margin, "folds": n,
                                "ci_method": "approximate_fold_t_interval"}))
        path = root / f"{metric}_90ci.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} configurations: {path}")
    print("90% CIs are descriptive fold-t intervals; folds share training data. No significance tests were performed.")


if __name__ == "__main__":
    main()
