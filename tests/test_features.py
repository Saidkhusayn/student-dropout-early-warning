"""Tests for derived features and the modeling sample."""

import math

import numpy as np
import pandas as pd

from dropout import config
from dropout.data import load_data, modeling_sample, semester_records_missing
from dropout.features import FEATURE_SETS, PIPELINE_FEATURES, ProgramRelativeGrade, add_trajectory_features


def make_students():
    # Student A: passed 3 of 6 in sem 1 (grade 12), all 6 in sem 2 (grade 13.5);
    #            admission 140 (= 14 on 0-20), prior qualification 130 (= 13);
    #            1 sem-1 unit with no exam taken
    # Student B: passed nothing in either semester (grade recorded as 0);
    #            enrolled in 6 units, then 4
    base = {
        "admission_grade": [140.0, 130.0], "previous_qualification_grade": [130.0, 120.0],
        "sem1_enrolled": [6, 6], "sem1_approved": [3, 0], "sem1_without_evaluations": [1, 3],
        "sem1_evaluations": [6, 3], "sem1_grade": [12.0, 0.0],
        "sem2_enrolled": [6, 4], "sem2_approved": [6, 0], "sem2_without_evaluations": [0, 4],
        "sem2_evaluations": [6, 0], "sem2_grade": [13.5, 0.0],
    }
    return add_trajectory_features(pd.DataFrame(base))


def test_pass_and_eval_rates():
    f = make_students()
    assert f.loc[0, "sem1_pass_rate"] == 0.5
    assert f.loc[0, "sem1_eval_rate"] == 1.0
    assert f.loc[1, "sem1_eval_rate"] == 0.5


def test_zero_grade_is_treated_as_missing():
    f = make_students()
    assert math.isnan(f.loc[1, "sem1_grade_real"])
    assert f.loc[1, "sem1_passed_none"] == 1


def test_vs_entry_and_changes():
    f = make_students()
    assert f.loc[0, "sem1_grade_vs_entry"] == 12.0 - 14.0
    assert f.loc[0, "sem2_grade_vs_entry"] == 13.5 - 14.0
    assert f.loc[0, "grade_change_s1_s2"] == 13.5 - 12.0
    assert f.loc[0, "pass_rate_change_s1_s2"] == 1.0 - 0.5
    assert f.loc[0, "sem2_passed_all"] == 1


def test_extended_features():
    f = make_students()
    assert f.loc[0, "sem1_skipped_exam_rate"] == 1 / 6
    assert f.loc[0, "sem1_exam_success_rate"] == 3 / 6
    assert f.loc[0, "sem1_grade_vs_prior"] == 12.0 - 13.0
    assert f.loc[0, "year1_pass_rate"] == 9 / 12
    assert f.loc[1, "load_change_s1_s2"] == -2
    assert math.isnan(f.loc[1, "sem2_exam_success_rate"])  # no exams taken


def test_no_stage_uses_future_data():
    for sets in FEATURE_SETS.values():
        assert sets["enrollment"] == []
        assert not any("sem2" in c or "s1_s2" in c or "year1" in c for c in sets["semester_1"])


def test_program_relative_grade_uses_training_data_only():
    train = pd.DataFrame({"course": [1] * 20 + [2] * 5, "sem1_grade_real": [12.0] * 20 + [16.0] * 5})
    test = pd.DataFrame({"course": [1, 2, 3], "sem1_grade_real": [14.0, 14.0, np.nan]})
    t = ProgramRelativeGrade(sems=(1,)).fit(train)
    out = t.transform(test)["sem1_grade_vs_program"]
    assert out[0] == 14.0 - 12.0              # course 1 average from training data
    assert out[1] == 14.0 - (20 * 12 + 5 * 16) / 25  # course 2 too small: overall average
    assert math.isnan(out[2])


def test_runs_on_real_data():
    f = add_trajectory_features(load_data())
    assert len(f) == 4424
    for col in FEATURE_SETS["extended"]["semester_2"]:
        assert col in f or col in PIPELINE_FEATURES
    assert f["sem1_pass_rate"].dropna().between(0, 1).all()


def test_missing_semester_records():
    df = load_data()
    missing = semester_records_missing(df)
    assert missing.sum() == 180
    assert (df.loc[missing, "course"] == 171).all()


def test_modeling_sample():
    df = load_data()
    enrollment = modeling_sample(df, "enrollment")
    semester = modeling_sample(df, "semester_1")
    assert set(enrollment[config.TARGET]) == {"Dropout", "Graduate"}
    assert len(enrollment) == 3630
    assert len(semester) == 3630 - 152  # 77 dropouts + 75 graduates with missing records
    assert not semester_records_missing(semester).any()
