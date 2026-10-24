"""Tests for trajectory features, using small hand-made students."""

import math

import pandas as pd

from dropout.data import load_data
from dropout.features import TRAJECTORY_BY_STAGE, add_trajectory_features


def make_students():
    # Student A: passed 3 of 6, grade 12, admitted with 140 (= 14 on the 0-20 scale)
    # Student B: enrolled in 0 units (special case)
    # Student C: enrolled in 6, passed none (grade recorded as 0)
    base = {
        "admission_grade": [140.0, 120.0, 130.0],
        "sem1_enrolled": [6, 0, 6], "sem1_approved": [3, 0, 0],
        "sem1_evaluations": [6, 0, 3], "sem1_grade": [12.0, 0.0, 0.0],
        "sem2_enrolled": [6, 0, 6], "sem2_approved": [6, 0, 0],
        "sem2_evaluations": [6, 0, 0], "sem2_grade": [13.5, 0.0, 0.0],
    }
    return add_trajectory_features(pd.DataFrame(base))


def test_pass_and_eval_rates():
    f = make_students()
    assert f.loc[0, "sem1_pass_rate"] == 0.5
    assert f.loc[0, "sem1_eval_rate"] == 1.0
    assert f.loc[2, "sem1_eval_rate"] == 0.5


def test_zero_enrolled_is_undefined_not_zero():
    f = make_students()
    assert math.isnan(f.loc[1, "sem1_pass_rate"])
    assert f.loc[1, "sem1_no_units"] == 1
    assert f.loc[1, "sem1_passed_none"] == 0


def test_zero_grade_is_treated_as_missing():
    f = make_students()
    assert math.isnan(f.loc[2, "sem1_grade_real"])
    assert f.loc[2, "sem1_passed_none"] == 1


def test_grade_gap_and_changes():
    f = make_students()
    assert f.loc[0, "sem1_grade_gap"] == 12.0 - 14.0
    assert f.loc[0, "grade_change"] == 13.5 - 12.0
    assert f.loc[0, "pass_rate_change"] == 1.0 - 0.5
    assert f.loc[0, "sem2_passed_all"] == 1


def test_no_stage_uses_future_data():
    assert all(not c.startswith("sem2_") for c in TRAJECTORY_BY_STAGE["semester_1"])
    assert TRAJECTORY_BY_STAGE["enrollment"] == []


def test_runs_on_real_data():
    f = add_trajectory_features(load_data())
    assert len(f) == 4424
    for col in TRAJECTORY_BY_STAGE["semester_2"]:
        assert col in f
    rates = f["sem1_pass_rate"].dropna()
    assert rates.between(0, 1).all()
