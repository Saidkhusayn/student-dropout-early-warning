# Early Detection of At-Risk Students through Performance-Trajectory Modeling

Undergraduate research project (UROP, Fall 2026), University of Minnesota
**Student researcher:** Saidkhusaynbek Bobojonov
**Faculty mentor:** Prof. Jaideep Srivastava, Department of Computer Science & Engineering

## Overview

Many students leave college before they finish their degree. This project asks **how early** a model can reliably flag students who are at risk of dropping out, so that an advisor still has time to help.

The key idea is to model each student's **trajectory**, meaning how their performance is changing compared to where they started, rather than a single snapshot. This idea comes from knowledge-tracing research.

### Research questions

1. Can we accurately flag students at risk of dropping out?
2. How much does accuracy improve as information arrives (at enrollment, after semester 1, after semester 2)?
3. How do classical models (logistic regression, random forest, gradient boosting) compare to neural networks?
4. Can we identify future dropouts among students who are still passing their classes?

## Data

This project uses the UCI [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89) dataset (CC BY 4.0). It covers 4,424 students, with 36 features and a target of Dropout, Enrolled or Graduate. See [`data/README.md`](data/README.md) for details.

## Repository structure

```
├── data/
│   ├── raw/              # original dataset, never modified
│   └── processed/        # generated files (git-ignored)
├── notebooks/            # numbered analysis notebooks
├── src/dropout/          # reusable code: config, data, features, modeling, plotting
├── scripts/              # experiment runners
├── tests/                # data, feature and modeling checks
├── results/
│   ├── figures/          # saved plots
│   └── tables/           # experiment results (CSV)
├── reports/              # write-ups and meeting summaries
├── requirements.txt
└── pyproject.toml
```

## Getting started

```bash
git clone https://github.com/<your-username>/student-dropout-early-warning.git
cd student-dropout-early-warning

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # makes `import dropout` work everywhere

pytest                           # data, feature and modeling checks
python scripts/run_baselines.py  # baseline experiments, ~5 min -> results/tables/
jupyter lab                      # notebooks/01 → 03
```

Quick check from Python:

```python
from dropout.data import load_data
df = load_data()
df.shape   # (4424, 37): 36 features + target
```

## Progress

- [x] Project setup, data loading and validation (`01_load_and_check.ipynb`)
- [x] Codebook for categorical codes (`data/codebook.csv`)
- [x] Exploratory data analysis (`02_eda.ipynb`, figures in `results/figures/`)
- [x] Trajectory feature engineering (`src/dropout/features.py`)
- [x] Baseline staged models: enrollment → semester 1 → semester 2 (`03_baseline_models.ipynb`)
- [ ] Hyperparameter tuning (nested CV)
- [ ] Neural network vs. classical models
- [ ] Feature importance and still-passing subgroup follow-up

## References

1. M. V. Martins, D. Tolledo, J. Machado, L. M. T. Baptista, and V. Realinho, "Early Prediction of Student's Performance in Higher Education: A Case Study," *WorldCIST 2021*, AISC vol. 1365, Springer, 2021, pp. 166–175.
2. V. Realinho, M. Vieira Martins, J. Machado, and L. Baptista, "Predict Students' Dropout and Academic Success" [Dataset], UCI Machine Learning Repository, 2021. https://doi.org/10.24432/C5MC89
3. M. V. Martins, L. Baptista, J. Machado, and V. Realinho, "Multi-Class Phased Prediction of Academic Performance and Dropout in Higher Education," *Applied Sciences* 13(8):4702, 2023. https://doi.org/10.3390/app13084702
4. S. Pandey and J. Srivastava, "RKT: Relation-Aware Self-Attention for Knowledge Tracing," *CIKM 2020*, pp. 1205–1214.

## License

The code is released under the [MIT License](LICENSE). The dataset is distributed under CC BY 4.0 by its original authors.
