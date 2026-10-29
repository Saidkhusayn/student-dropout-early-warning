"""Tests for feature sets and the CV pipeline."""

import numpy as np

from dropout import config
from dropout.data import load_data, modeling_sample
from dropout.features import add_trajectory_features
from dropout.modeling import MODELS, cross_validate, feature_columns, subgroup_recall, subgroup_scores, target


def test_enrollment_features_have_no_semester_data():
    cols = feature_columns("enrollment", "extended", financial=True)
    assert not any(c.startswith(("sem1_", "sem2_")) for c in cols)


def test_semester_1_has_no_semester_2_data():
    cols = feature_columns("semester_1", "extended", financial=True)
    assert not any("sem2" in c or "s1_s2" in c or "year1" in c for c in cols)


def test_financial_switch():
    strict = feature_columns("enrollment")
    full = feature_columns("enrollment", financial=True)
    assert set(full) - set(strict) == set(config.TIMING_UNCLEAR_FEATURES)


def test_same_seed_same_folds():
    df = modeling_sample(add_trajectory_features(load_data()), "semester_1").sample(400, random_state=0)
    cols = feature_columns("semester_1", "extended")
    a, _ = cross_validate(df, cols, "logreg", n_repeats=1)
    b, _ = cross_validate(df, cols, "logreg", n_repeats=1)
    assert a.equals(b)


def test_every_model_runs_and_covers_every_row():
    df = modeling_sample(add_trajectory_features(load_data()), "semester_2").sample(300, random_state=0)
    cols = feature_columns("semester_2", "extended")
    for model in MODELS:
        folds, oof = cross_validate(df, cols, model, n_splits=3, n_repeats=1)
        assert len(folds) == 3
        assert not np.isnan(oof).any()
        assert folds["roc_auc"].between(0, 1).all()


def test_subgroup_recall():
    y = np.array([1, 1, 0, 1])
    oof = np.array([[0.9, 0.2, 0.9, 0.8]])
    mask = np.array([True, True, True, False])
    assert subgroup_recall(oof, y, mask) == 0.5


def test_target_is_dropout():
    df = modeling_sample(load_data(), "enrollment")
    assert target(df).mean() == (df[config.TARGET] == "Dropout").mean()


def test_subgroup_scores():
    y = np.array([1, 0, 0, 1, 0])
    oof = np.array([[0.9, 0.6, 0.1, 0.2, 0.9]])
    mask = np.array([True, True, True, False, False])
    out = subgroup_scores(oof, y, mask)
    assert out["n"] == 3 and out["n_dropouts"] == 1
    assert out["recall"] == 1.0
    assert out["false_alarm_rate"] == 0.5
    assert out["roc_auc"] == 1.0
