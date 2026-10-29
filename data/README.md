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
- The file is `;`-separated and UTF-8 with a byte-order mark. It has no empty cells, but some zeros stand for missing values (see [Data notes](#data-notes)).

## Terminology

The data come from a Portuguese institution. Several terms differ from US usage.

| Term in the data | Meaning |
|---|---|
| Course | Degree program (major), recorded at admission |
| Curricular unit | An individual class taken in a semester |
| Admission grade | National admission score, 0–200: secondary-school grades combined with national entrance exams |
| Semester grade | Average on a 0–20 scale over the units **passed** that semester; failed units are not included |
| Evaluations | Exam attempts in a semester, including retakes |
| Credited | Units recognized from prior study |
| Enrolled (target) | Still studying when the data were collected; final outcome unknown |
| Debtor / Tuition fees up to date | Yes/no financial flags; not defined by the authors |

## Codebook

`codebook.csv` translates every coded value into its meaning. It has 261 rows with the columns `variable` (the snake_case column name), `code` and `meaning`. It covers the 9 categorical columns and the 8 yes/no columns.

- **Sources:** the variable descriptions on the [UCI dataset page](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) and Appendix Tables A1–A7 of Realinho et al., "Predicting Student Dropout and Academic Success," *Data* 7(11):146, 2022 ([link](https://www.mdpi.com/2306-5729/7/11/146)).
- **The paper's numbering is different.** The paper numbers each list 1, 2, 3, …, but the CSV uses the original UCI codes (e.g. course `9254` = Tourism). The paper's lists are in the same order as the sorted UCI codes, and spot checks against the UCI metadata confirm the mapping.
- **Every code in the data is covered** (checked by `tests/test_data.py`).
- **In code:** `dropout.data.decode(df)` swaps codes for labels. Use it for tables and plots only; models keep the numeric codes.

## Data notes

Gaps and exceptions found during exploration (`notebooks/02_eda.ipynb`). Each one affects how the data are used.

1. **First year only.** Class-level data cover semesters 1 and 2. The outcome (Dropout / Enrolled / Graduate) was recorded later. Nothing is available for later years.
2. **Enrolled students have no final outcome.** Models use Dropout vs. Graduate only (n = 3,630).
3. **A semester grade of 0 means no unit was passed,** not an average of 0. It is treated as missing.
4. **Missing semester records for 180 students.** Every semester column is 0 in both semesters, including credited units, yet 75 of these students graduated. All 180 are in course 171 (Animation and Multimedia Design, 180 of its 215 students). They are kept at the enrollment stage and removed from the semester stages (`dropout.data.semester_records_missing`).
5. **Tuition and debtor status: timing unknown.** Neither variable is defined by the authors, and no paper states when they were recorded. Martins et al. (2023) place both at enrollment. The earlier, enrollment-only version of the data (Martins et al., 2021) lists student debt among variables observable before registration but does not include tuition status, which first appears in the public dataset. They are the strongest enrollment-stage signals, so every stage is reported with and without them.
6. **Debtor and tuition status are different variables.** They overlap (246 students are both), but 282 students are behind on tuition without being debtors, and 257 debtors are up to date on tuition.
7. **Evaluations include retakes,** so evaluations per enrolled unit can exceed 1. It is not an attendance measure.
8. **Code numbering differs between sources.** The appendix of the 2022 data paper numbers categories 1, 2, 3, …; the CSV uses the original UCI codes. See [Codebook](#codebook).

## Notes

- **Never edit `raw/`.** Anything derived from it goes in `processed/`, which git ignores and the code re-creates.
- `src/dropout/config.py` maps the original column names to snake_case. The mapping also fixes the source's `Nacionality` typo and a stray tab in `Daytime/evening attendance`.
- Several integer columns are category codes, not quantities (e.g. `Course`, `Application mode`, parents' occupation). See `codebook.csv`.

## References

- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., and Realinho, V. "Early Prediction of Student's Performance in Higher Education: A Case Study." *WorldCIST 2021*, AISC 1365, pp. 166–175. https://doi.org/10.1007/978-3-030-72657-7_16
- Realinho, V., Machado, J., Baptista, L., and Martins, M. V. "Predicting Student Dropout and Academic Success." *Data* 7(11):146, 2022. https://www.mdpi.com/2306-5729/7/11/146
- Martins, M. V., Baptista, L., Machado, J., and Realinho, V. "Multi-Class Phased Prediction of Academic Performance and Dropout in Higher Education." *Applied Sciences* 13(8):4702, 2023. https://www.mdpi.com/2076-3417/13/8/4702
