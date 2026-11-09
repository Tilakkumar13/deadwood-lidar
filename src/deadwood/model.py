"""Baseline classifier and honest evaluation.

Two decisions here are methodological rather than technical, and both change
what the numbers mean:

1. Grouped cross-validation. Trees in the same plot share a flight line,
   sensor settings, stand structure and often species. Random k-fold puts
   neighbours of a training tree into the test set and inflates the score,
   sometimes badly. Splitting by plot measures what you actually care about:
   does this transfer to a stand the model has not seen.

2. Feature-group ablation. If the model leans on intensity, the result will
   not survive a different sensor or acquisition. Scoring each group on its
   own tells you whether the signal is structural (transferable) or
   radiometric (not), which is a thing a reviewer will ask about.

Start with gradient boosting, not a point-cloud network. With a few thousand
stems and 23 engineered features, a tree ensemble is the right tool, and it
gives you feature importances to argue from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .features import ALL_FEATURES, FEATURE_GROUPS


def make_model(random_state: int = 0) -> HistGradientBoostingClassifier:
    """Handles NaNs natively, which matters: sparse stems legitimately produce
    NaN shape features and imputing them would invent structure."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        # "auto" turns early stopping on only above 10k samples. Forcing it
        # True carves a validation split out of small training folds, which
        # can leave the model unable to fit and silently scoring at chance.
        early_stopping="auto",
        validation_fraction=0.15,
        random_state=random_state,
    )


def grouped_cv_predict(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """Out-of-fold probabilities under GroupKFold (groups = plot IDs)."""
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("need at least 2 distinct plots for grouped CV")
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
    return oof


def evaluate(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Metrics that suit an imbalanced problem.

    Dead stems are the minority class, so accuracy is close to meaningless -
    predicting 'all live' can score 90%. Average precision and per-class
    recall are what to report.
    """
    pred = (proba >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(
            y, pred, target_names=["live", "dead"], output_dict=True, zero_division=0
        ),
    }


def ablate_feature_groups(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    """Score each feature group alone, and all groups together.

    Read it like this: if 'intensity' alone scores near the full model, the
    result is radiometric and will not transfer across sensors. If 'returns'
    and 'shape' carry it, the signal is structural and far more robust.
    """
    rows = []
    for name, cols in {**FEATURE_GROUPS, "ALL": ALL_FEATURES}.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        proba = grouped_cv_predict(X[cols], y, groups)
        m = evaluate(y, proba)
        rows.append(
            {
                "features": name,
                "n_features": len(cols),
                "roc_auc": round(m["roc_auc"], 4),
                "average_precision": round(m["average_precision"], 4),
                "dead_recall": round(m["report"]["dead"]["recall"], 4),
            }
        )
    return pd.DataFrame(rows).sort_values("average_precision", ascending=False)
