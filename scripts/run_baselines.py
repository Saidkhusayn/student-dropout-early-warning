"""Run the baseline experiments and save result tables.

    python scripts/run_baselines.py

Experiments (Dropout vs. Graduate, 5-fold CV repeated 3 times, 3 models each):
  stage_curve   3 stages, raw dataset columns only
  features      semester stages with the core and extended derived features
  financial     every stage with debtor / tuition status added
  full_sample   enrollment stage on all 3,630 students (incl. missing semester records)

All experiments except full_sample use the same 3,478 students, so they share
CV folds and can be compared fold by fold.

Outputs in results/tables/:
  cv_folds.csv        one row per experiment x model x fold
  subgroup.csv        metrics within students who passed every semester-1 unit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dropout import config  # noqa: E402
from dropout.data import load_data, modeling_sample  # noqa: E402
from dropout.features import add_trajectory_features  # noqa: E402
from dropout.modeling import MODELS, cross_validate, feature_columns, subgroup_scores, target  # noqa: E402

TABLES = config.PROJECT_ROOT / "results" / "tables"

SEMESTERS = ("semester_1", "semester_2")

# (experiment, stage, feature_set, financial, sample)
EXPERIMENTS = (
    [("stage_curve", s, "raw", False, "common") for s in config.STAGES]
    + [("features", s, fs, False, "common") for fs in ("core", "extended") for s in SEMESTERS]
    + [("financial", s, "raw", True, "common") for s in config.STAGES]
    + [("full_sample", "enrollment", "raw", fin, "full") for fin in (False, True)]
)


def main() -> None:
    data = add_trajectory_features(load_data())
    samples = {
        "common": modeling_sample(data, "semester_1"),   # 3,478 students with semester records
        "full": modeling_sample(data, "enrollment"),      # 3,630 students
    }
    still_passing = samples["common"]["sem1_passed_all"].to_numpy() == 1
    y_common = target(samples["common"])

    folds, subgroup = [], []
    for experiment, stage, feature_set, financial, sample in EXPERIMENTS:
        df = samples[sample]
        cols = feature_columns(stage, feature_set, financial)
        for model in MODELS:
            start = time.time()
            scores, oof = cross_validate(df, cols, model)
            tags = {"experiment": experiment, "stage": stage, "feature_set": feature_set,
                    "financial": financial, "sample": sample, "n_students": len(df),
                    "n_features": len(cols), "model": model}
            folds.append(scores.assign(**tags))
            if sample == "common":
                subgroup.append({**tags, **subgroup_scores(oof, y_common, still_passing)})
            print(f"{experiment:12s} {stage:11s} {feature_set:9s} fin={financial!s:5s} {model:13s} "
                  f"PR-AUC {scores['pr_auc'].mean():.3f}  ({time.time() - start:.0f}s)", flush=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    tag_cols = ["experiment", "stage", "feature_set", "financial", "sample", "n_students", "n_features", "model"]
    out = pd.concat(folds, ignore_index=True)
    out[tag_cols + [c for c in out if c not in tag_cols]].to_csv(TABLES / "cv_folds.csv", index=False)
    pd.DataFrame(subgroup).round(4).to_csv(TABLES / "subgroup.csv", index=False)
    print(f"saved to {TABLES}")


if __name__ == "__main__":
    main()
