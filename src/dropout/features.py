"""Trajectory features: how a student's performance is CHANGING, not just where it is.

Every feature here uses only information available at its stage, so a
semester-1 feature never looks at semester-2 data.

Undefined values are left as NaN on purpose (e.g. pass rate for a student
enrolled in 0 units). The modeling pipeline decides how to fill them; tree
models such as XGBoost can use NaN directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Admission grade is on a 0-200 scale; semester grades are on 0-20.
ADMISSION_TO_GRADE_SCALE = 10.0


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, with NaN where the denominator is 0."""
    return numerator / denominator.where(denominator > 0)


def semester_features(df: pd.DataFrame, sem: int) -> pd.DataFrame:
    """Per-semester features for semester ``sem`` (1 or 2)."""
    p = f"sem{sem}_"
    enrolled = df[p + "enrolled"]
    approved = df[p + "approved"]

    # In the raw data a grade of 0 means "no unit approved", not a real average.
    real_grade = df[p + "grade"].where(approved > 0)

    return pd.DataFrame(
        {
            p + "no_units": (enrolled == 0).astype(int),
            p + "passed_none": ((enrolled > 0) & (approved == 0)).astype(int),
            p + "passed_all": ((enrolled > 0) & (approved == enrolled)).astype(int),
            p + "pass_rate": _safe_ratio(approved, enrolled),
            p + "eval_rate": _safe_ratio(df[p + "evaluations"], enrolled),
            p + "grade_real": real_grade,
            p + "grade_gap": real_grade - df["admission_grade"] / ADMISSION_TO_GRADE_SCALE,
        },
        index=df.index,
    )


def change_features(sem1: pd.DataFrame, sem2: pd.DataFrame) -> pd.DataFrame:
    """Semester 1 -> semester 2 changes (positive = improving)."""
    return pd.DataFrame(
        {
            "grade_change": sem2["sem2_grade_real"] - sem1["sem1_grade_real"],
            "pass_rate_change": sem2["sem2_pass_rate"] - sem1["sem1_pass_rate"],
        },
        index=sem1.index,
    )


def add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every trajectory feature added."""
    s1 = semester_features(df, 1)
    s2 = semester_features(df, 2)
    return pd.concat([df, s1, s2, change_features(s1, s2)], axis=1)


SEM1_TRAJECTORY = [
    "sem1_no_units", "sem1_passed_none", "sem1_passed_all",
    "sem1_pass_rate", "sem1_eval_rate", "sem1_grade_real", "sem1_grade_gap",
]
SEM2_TRAJECTORY = [c.replace("sem1_", "sem2_") for c in SEM1_TRAJECTORY]
CHANGE_TRAJECTORY = ["grade_change", "pass_rate_change"]

# New features available at each modeling stage (on top of config.STAGES).
TRAJECTORY_BY_STAGE = {
    "enrollment": [],
    "semester_1": SEM1_TRAJECTORY,
    "semester_2": SEM1_TRAJECTORY + SEM2_TRAJECTORY + CHANGE_TRAJECTORY,
}
