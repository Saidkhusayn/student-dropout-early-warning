# Data

## Source

`raw/data.csv` is the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

> V. Realinho, M. Vieira Martins, J. Machado, and L. Baptista. *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository, 2021. https://doi.org/10.24432/C5MC89

- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is redistributed here unchanged, with attribution.
- **Official download:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **SHA-256:** `19737f02127e85ef29af536e46888a7721b6c530c3e1c6aedfbb09109c030a71`

## Contents

- 4,424 students from a Portuguese higher-education institution, each in one row.
- 37 columns: 36 features plus `Target` (`Dropout` / `Enrolled` / `Graduate`).
- The file is `;`-separated and UTF-8 with a byte-order mark. It has no missing values.

## Codebook

`codebook.csv` translates every coded value into its meaning. It has 261 rows with the columns `variable` (the snake_case column name), `code` and `meaning`. It covers the 9 categorical columns and the 8 yes/no columns.

- **Sources:** the variable descriptions on the [UCI dataset page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and Appendix Tables A1–A7 of Realinho et al., "Predicting Student Dropout and Academic Success," *Data* 7(11):146, 2022 ([link](https://www.mdpi.com/2306-5729/7/11/146)).
- **The paper's numbering is different.** The paper numbers each list 1, 2, 3, …, but the CSV uses the original UCI codes (e.g. course `9254` = Tourism). The paper's lists are in the same order as the sorted UCI codes, and spot checks against the UCI metadata confirm the mapping.
- **Every code in the data is covered** (checked by `tests/test_data.py`).
- **In code:** `dropout.data.decode(df)` swaps codes for labels. Use it for tables and plots only; models keep the numeric codes.

## Notes

- **Never edit `raw/`.** Anything derived from it goes in `processed/`, which git ignores and the code re-creates.
- `src/dropout/config.py` maps the original column names to snake_case. The mapping also fixes the source's `Nacionality` typo and a stray tab in `Daytime/evening attendance`.
- Several integer columns are category codes, not quantities (e.g. `Course`, `Application mode`, parents' occupation). The UCI page has the code tables.
