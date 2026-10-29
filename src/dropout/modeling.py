"""Feature sets, model pipelines and cross-validation for the staged models.

All preprocessing (encoding, imputation, scaling) lives inside the pipeline, so
it is fit on the training folds only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dropout import config
from dropout.features import FEATURE_SETS, PIPELINE_FEATURES, ProgramRelativeGrade

MODELS = ("logreg", "random_forest", "gboost")
THRESHOLD = 0.5


def feature_columns(stage: str, feature_set: str = "raw", financial: bool = False) -> list[str]:
    """Columns a model sees at ``stage``.

    feature_set: "raw" (dataset columns only), "core" or "extended" (see features.py).
    financial: keep debtor / tuition status (timing unclear, see data/README.md).
    """
    cols = list(config.STAGES[stage])
    if not financial:
        cols = [c for c in cols if c not in config.TIMING_UNCLEAR_FEATURES]
    return cols + FEATURE_SETS[feature_set][stage]


def input_columns(columns: list[str]) -> list[str]:
    """Columns taken from the data frame; the rest are built inside the pipeline."""
    return [c for c in columns if c not in PIPELINE_FEATURES]


def target(df: pd.DataFrame) -> np.ndarray:
    """1 = Dropout, 0 = Graduate."""
    return (df[config.TARGET] == config.POSITIVE_CLASS).astype(int).to_numpy()


def make_pipeline(model: str, columns: list[str], seed: int = config.RANDOM_SEED) -> Pipeline:
    """Preprocessing + classifier for one of ``MODELS``."""
    categorical = [c for c in columns if c in config.CATEGORICAL_FEATURES]
    numeric = [c for c in columns if c not in categorical]
    # Rare categories (< 20 students) are pooled into one "infrequent" column.
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)

    if model == "gboost":
        numeric_steps = "passthrough"  # handles missing values natively
    else:
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if model == "logreg":
            steps.append(("scale", StandardScaler()))
        numeric_steps = Pipeline(steps)

    pre = ColumnTransformer([("cat", onehot, categorical), ("num", numeric_steps, numeric)])

    if model == "logreg":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    elif model == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
    elif model == "gboost":
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=seed)
    else:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    steps = [("pre", pre), ("clf", clf)]
    sems = tuple(int(c[3]) for c in columns if c in PIPELINE_FEATURES)
    if sems:
        steps.insert(0, ("program", ProgramRelativeGrade(sems=sems)))
    return Pipeline(steps)


def scores(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Threshold-free and thresholded metrics for the Dropout class."""
    pred = (prob >= THRESHOLD).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, prob),
        "roc_auc": roc_auc_score(y_true, prob),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
    }


def cross_validate(df: pd.DataFrame, columns: list[str], model: str, n_splits: int = 5,
                   n_repeats: int = 3, seed: int = config.RANDOM_SEED):
    """Repeated stratified k-fold CV.

    The same ``df`` and ``seed`` always give the same folds, so different models
    and feature sets evaluated on one sample can be compared fold by fold.

    Returns (fold_scores, oof) where oof[r, i] is the out-of-fold Dropout
    probability for row i in repeat r.
    """
    X, y = df[input_columns(columns)], target(df)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.full((n_repeats, len(df)), np.nan)
    rows = []
    for k, (train, test) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(k, n_splits)
        pipe = make_pipeline(model, columns, seed).fit(X.iloc[train], y[train])
        prob = pipe.predict_proba(X.iloc[test])[:, 1]
        oof[repeat, test] = prob
        rows.append({"repeat": repeat, "fold": fold, **scores(y[test], prob)})
    return pd.DataFrame(rows), oof


def subgroup_recall(oof: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Recall among dropouts in ``mask``, averaged over CV repeats."""
    idx = mask & (y == 1)
    return float(np.mean([(p[idx] >= THRESHOLD).mean() for p in oof]))


def subgroup_scores(oof: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Metrics within a subgroup, averaged over CV repeats.

    Recall alone can mislead: a model that flags everyone in the group has high
    recall. ``false_alarm_rate`` is the share of graduates in the group flagged,
    and the AUCs measure whether dropouts are ranked above graduates there.
    """
    y_sub = y[mask]
    per_repeat = [scores(y_sub, p[mask]) | {"false_alarm_rate": float((p[mask][y_sub == 0] >= THRESHOLD).mean())}
                  for p in oof]
    out = pd.DataFrame(per_repeat).mean().to_dict()
    out["n"], out["n_dropouts"] = int(mask.sum()), int(y_sub.sum())
    return out
