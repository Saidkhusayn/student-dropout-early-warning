"""Project-wide constants: paths, column names, and feature groups by stage."""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
CODEBOOK_PATH = DATA_DIR / "codebook.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# SHA-256 of the raw UCI file, so we can tell if it was ever changed.
RAW_DATA_SHA256 = "19737f02127e85ef29af536e46888a7721b6c530c3e1c6aedfbb09109c030a71"

# ---------------------------------------------------------------- shape
EXPECTED_N_ROWS = 4424
EXPECTED_N_COLS = 37  # 36 features + 1 target
RANDOM_SEED = 42

# ---------------------------------------------------------------- columns
# Original UCI header -> snake_case name used everywhere in this project.
# (The raw file also spells "Nationality" as "Nacionality" and has a stray
# tab after "Daytime/evening attendance"; both are fixed here.)
COLUMN_RENAMES = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime_attendance",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality",
    "Mother's qualification": "mother_qualification",
    "Father's qualification": "father_qualification",
    "Mother's occupation": "mother_occupation",
    "Father's occupation": "father_occupation",
    "Admission grade": "admission_grade",
    "Displaced": "displaced",
    "Educational special needs": "special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_up_to_date",
    "Gender": "gender",
    "Scholarship holder": "scholarship_holder",
    "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "sem1_credited",
    "Curricular units 1st sem (enrolled)": "sem1_enrolled",
    "Curricular units 1st sem (evaluations)": "sem1_evaluations",
    "Curricular units 1st sem (approved)": "sem1_approved",
    "Curricular units 1st sem (grade)": "sem1_grade",
    "Curricular units 1st sem (without evaluations)": "sem1_without_evaluations",
    "Curricular units 2nd sem (credited)": "sem2_credited",
    "Curricular units 2nd sem (enrolled)": "sem2_enrolled",
    "Curricular units 2nd sem (evaluations)": "sem2_evaluations",
    "Curricular units 2nd sem (approved)": "sem2_approved",
    "Curricular units 2nd sem (grade)": "sem2_grade",
    "Curricular units 2nd sem (without evaluations)": "sem2_without_evaluations",
    "Unemployment rate": "unemployment_rate",
    "Inflation rate": "inflation_rate",
    "GDP": "gdp",
    "Target": "target",
}

TARGET = "target"
TARGET_CLASSES = ("Dropout", "Enrolled", "Graduate")

# Models predict Dropout vs. Graduate. "Enrolled" students have no final outcome yet.
MODEL_CLASSES = ("Dropout", "Graduate")
POSITIVE_CLASS = "Dropout"

# Strong signals whose recording date is not stated by the dataset authors.
# The enrollment stage is reported with and without them (possible leakage).
TIMING_UNCLEAR_FEATURES = ["debtor", "tuition_up_to_date"]

# Integer codes that stand for categories, not quantities -> one-hot encode.
CATEGORICAL_FEATURES = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nationality",
    "mother_qualification",
    "father_qualification",
    "mother_occupation",
    "father_occupation",
]

# ---------------------------------------------------------------- stages
# Features grouped by WHEN a school would actually have them.
SEM1_FEATURES = [c for c in COLUMN_RENAMES.values() if c.startswith("sem1_")]
SEM2_FEATURES = [c for c in COLUMN_RENAMES.values() if c.startswith("sem2_")]
ENROLLMENT_FEATURES = [
    c for c in COLUMN_RENAMES.values()
    if c not in SEM1_FEATURES + SEM2_FEATURES + [TARGET]
]

STAGES = {
    "enrollment": ENROLLMENT_FEATURES,
    "semester_1": ENROLLMENT_FEATURES + SEM1_FEATURES,
    "semester_2": ENROLLMENT_FEATURES + SEM1_FEATURES + SEM2_FEATURES,
}
