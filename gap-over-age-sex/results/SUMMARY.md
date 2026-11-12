# ECG-age gap over age and sex on CODE-15%: results

Generated from `results/results.json` by `make_summary.py` (1000 bootstrap samples, seed 20261108).

Cohort: 233,647 patients (one row each, first exam), 8,341 deaths, median follow-up 3.46 years. Split 80/20 by patient: held-out 46,730 patients, 1,721 deaths. Cox models fitted on the training patients; Harrell's C on the held-out patients.

| Model | Held-out C (95% CI) | 5-fold CV mean (SD) |
|---|---|---|
| Age + sex | 0.790 (0.780 to 0.801) | 0.795 (0.007) |
| Age + sex + ECG-age gap | 0.799 (0.789 to 0.809) | 0.803 (0.007) |
| Age + sex + corrected gap | 0.799 (identical by construction) | |
| Corrected gap alone | 0.594 (0.579 to 0.608) | 0.586 (0.008) |
| Raw gap alone | 0.535 (0.520 to 0.551) | 0.542 (0.008) |

- Adding the gap to age and sex: +0.009 (95% CI 0.006 to 0.012).
- Hazard ratio per year of gap, adjusted for age and sex: 1.0288 (raw) and 1.0288 (corrected): the same, as expected.
- Unadjusted, the raw gap's hazard ratio is 0.9863 per year (below 1, the wrong direction) and the corrected gap's is 1.0240. The correction lifts the gap-alone C by +0.058 (95% CI 0.028 to 0.088).

Gain from adding the gap to age + sex, within subgroups of the held-out patients:

| Subgroup | Patients | Deaths | Age + sex | + gap | Gain (95% CI) |
|---|---|---|---|---|---|
| male | 18,965 | 906 | 0.765 | 0.777 | +0.012 (0.008 to 0.017) |
| female | 27,765 | 815 | 0.800 | 0.807 | +0.006 (0.002 to 0.011) |
| age 16-39 | 15,807 | 85 | 0.618 | 0.688 | +0.070 (0.030 to 0.112) |
| age 40-59 | 13,613 | 239 | 0.618 | 0.675 | +0.057 (0.036 to 0.082) |
| age 60-74 | 10,181 | 523 | 0.625 | 0.646 | +0.021 (0.006 to 0.038) |
| age 75-+ | 7,129 | 874 | 0.571 | 0.589 | +0.018 (0.004 to 0.031) |

Checks: lifelines' C against a written-out Harrell's C on 2,500 held-out patients: age_sex 0.7909 vs 0.7909; age_sex_gap 0.8036 vs 0.8037; gap_raw 0.5440 vs 0.5440; gap_bc 0.5966 vs 0.5966.
Raw and corrected models with age: risk-score correlation 1.000000, C difference 0.000000.
