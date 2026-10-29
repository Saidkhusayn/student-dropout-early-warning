"""Derived features describing how a student is doing, not just raw counts.

Two groups, so each can be tested separately:

core      pass rates, real grades, grade relative to entry level, and the
          semester 1 -> 2 changes (the trajectory hypothesis)
extended  skipped-exam rate, exam success rate, grade relative to the prior
          qualification, course-load change, first-year pass rate, and grade
          relative to program peers

Every feature uses only information available at its stage, so a semester-1
feature never looks at semester-2 data.

Naming: "_vs_entry" / "_vs_prior" / "_vs_program" compare a semester grade with
the student's admission grade, prior-qualification grade, or program average;
"_s1_s2" is the change from semester 1 to semester 2 (positive = improving).

Undefined values are left as NaN (e.g. a grade when no unit was passed); the
modeling pipeline fills them. Students with missing semester records are removed
before semester-stage modeling (``dropout.data.semester_records_missing``).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Admission and prior-qualification grades are on 0-200; semester grades on 0-20.
GRADE_SCALE = 10.0


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, with NaN where the denominator is 0."""
    return numerator / denominator.where(denominator > 0)


def semester_features(df: pd.DataFrame, sem: int) -> pd.DataFrame:
    """Per-semester features for semester ``sem`` (1 or 2)."""
    p = f"sem{sem}_"
    enrolled, approved = df[p + "enrolled"], df[p + "approved"]
    evaluations = df[p + "evaluations"]

    # In the raw data a grade of 0 means "no unit approved", not a real average.
    real_grade = df[p + "grade"].where(approved > 0)

    return pd.DataFrame(
        {
            # core
            p + "passed_none": ((enrolled > 0) & (approved == 0)).astype(int),
            p + "passed_all": ((enrolled > 0) & (approved == enrolled)).astype(int),
            p + "pass_rate": _safe_ratio(approved, enrolled),
            p + "eval_rate": _safe_ratio(evaluations, enrolled),
            p + "grade_real": real_grade,
            p + "grade_vs_entry": real_grade - df["admission_grade"] / GRADE_SCALE,
            # extended
            p + "skipped_exam_rate": _safe_ratio(df[p + "without_evaluations"], enrolled),
            p + "exam_success_rate": _safe_ratio(approved, evaluations),
            p + "grade_vs_prior": real_grade - df["previous_qualification_grade"] / GRADE_SCALE,
        },
        index=df.index,
    )


def change_features(df: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame) -> pd.DataFrame:
    """Semester 1 -> semester 2 changes and first-year totals."""
    return pd.DataFrame(
        {
            # core
            "grade_change_s1_s2": s2["sem2_grade_real"] - s1["sem1_grade_real"],
            "pass_rate_change_s1_s2": s2["sem2_pass_rate"] - s1["sem1_pass_rate"],
            # extended
            "load_change_s1_s2": df["sem2_enrolled"] - df["sem1_enrolled"],
            "year1_pass_rate": _safe_ratio(df["sem1_approved"] + df["sem2_approved"],
                                           df["sem1_enrolled"] + df["sem2_enrolled"]),
        },
        index=df.index,
    )


def add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every row-level derived feature added.

    The program-relative grades are not added here: they use averages over
    other students, so they are computed inside the model pipeline from the
    training fold only (``ProgramRelativeGrade``).
    """
    s1, s2 = semester_features(df, 1), semester_features(df, 2)
    return pd.concat([df, s1, s2, change_features(df, s1, s2)], axis=1)


class ProgramRelativeGrade(BaseEstimator, TransformerMixin):
    """Add ``sem*_grade_vs_program``: real grade minus the program's average grade.

    Program averages are learned in ``fit`` (training fold only). Programs with
    fewer than ``min_count`` graded students fall back to the overall average.
    """

    def __init__(self, sems: tuple[int, ...] = (1, 2), min_count: int = 20):
        self.sems = sems
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y=None):
        self.means_, self.overall_ = {}, {}
        for s in self.sems:
            grades = X[f"sem{s}_grade_real"]
            stats = grades.groupby(X["course"]).agg(["mean", "count"])
            self.means_[s] = stats.loc[stats["count"] >= self.min_count, "mean"]
            self.overall_[s] = grades.mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for s in self.sems:
            mean = X["course"].map(self.means_[s]).fillna(self.overall_[s])
            X[f"sem{s}_grade_vs_program"] = X[f"sem{s}_grade_real"] - mean
        return X


def _sem(names: list[str], s: int) -> list[str]:
    return [f"sem{s}_{n}" for n in names]


CORE_SEM = ["passed_none", "passed_all", "pass_rate", "eval_rate", "grade_real", "grade_vs_entry"]
EXTENDED_SEM = ["skipped_exam_rate", "exam_success_rate", "grade_vs_prior", "grade_vs_program"]

CORE = {
    "enrollment": [],
    "semester_1": _sem(CORE_SEM, 1),
    "semester_2": _sem(CORE_SEM, 1) + _sem(CORE_SEM, 2) + ["grade_change_s1_s2", "pass_rate_change_s1_s2"],
}
EXTENDED = {
    "enrollment": [],
    "semester_1": _sem(EXTENDED_SEM, 1),
    "semester_2": _sem(EXTENDED_SEM, 1) + _sem(EXTENDED_SEM, 2) + ["load_change_s1_s2", "year1_pass_rate"],
}

# Derived features added on top of the raw columns, by feature set and stage.
FEATURE_SETS = {
    "raw": {stage: [] for stage in CORE},
    "core": CORE,
    "extended": {stage: CORE[stage] + EXTENDED[stage] for stage in CORE},
}

# Features computed inside the pipeline (they need training-fold statistics).
PIPELINE_FEATURES = ["sem1_grade_vs_program", "sem2_grade_vs_program"]
