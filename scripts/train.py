#!/usr/bin/env python3
"""Train and evaluate on a prepared feature table.

    python scripts/train.py data/features.parquet
"""

import sys
from pathlib import Path

import pandas as pd

from deadwood.features import ALL_FEATURES
from deadwood.model import ablate_feature_groups, evaluate, grouped_cv_predict


def main(path: str) -> None:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    X = df[[c for c in ALL_FEATURES if c in df.columns]]
    y = df["is_dead"].to_numpy()
    groups = df["plotID"].to_numpy()

    print(f"{len(df)} stems, {y.mean():.1%} dead, {len(set(groups))} plots\n")

    proba = grouped_cv_predict(X, y, groups)
    m = evaluate(y, proba)
    print(f"ROC AUC           {m['roc_auc']:.3f}")
    print(f"Average precision {m['average_precision']:.3f}")
    print(f"Dead recall       {m['report']['dead']['recall']:.3f}")
    print(f"Dead precision    {m['report']['dead']['precision']:.3f}")
    print(f"\nConfusion matrix (rows true, cols pred):\n{m['confusion_matrix']}")

    print("\nFeature-group ablation:")
    print(ablate_feature_groups(X, y, groups).to_string(index=False))
    print(
        "\nIf 'intensity' alone approaches 'ALL', the model is leaning on "
        "radiometry and will not transfer to another sensor."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
