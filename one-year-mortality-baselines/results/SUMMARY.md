# One-year mortality on CODE-15%: the rows under a foundation model

Generated from `results/results.json` by `make_summary.py` (1000 bootstrap resamples, seed base 20261114). Every number below was computed in this repository (TESTED) unless it sits under "Published", where it is copied from the paper or its source data.

## Cohort

- 233,647 patients with a death flag and a follow-up time in `exams.csv`, one row each.
- 2,812 died within one year; 222,057 were followed for at least one year; 8,778 were alive with under one year of follow-up.
- Main analysis: the 8,778 are left out. Death rate 1.25%. Keeping them as survivors (sensitivity A) gives 1.20%. The paper reports 1.23%, which sits between the two.
- Split by patient, 80/10/10, stratified on the label, five times (splits 0 to 4). Split 0: 179,868 train (2,250 deaths), 22,521 validation (281 deaths), 22,480 test (281 deaths).
- Network features come from the exam the outcome is attached to, for every patient: 217,932 from the saved features of each patient's lowest exam id, and 15,715 extracted from the outcome exam's trace by `extract_outcome_exams.py` (its predicted ages match `nn_predicted_age` in `exams.csv` to 0.0005 years). An earlier version left those 15,715 patients out; `DATA_EXCLUSIONS.md` says why and what changed.

## The ladder

Every test patient of split 0: 22,480 patients, 281 deaths. All six rows are scored on the same patients, so the differences are paired.

| Row | Inputs | Test AUC, split 0 (95% CI) | Mean of 5 splits (range) | Difference from row 2, split 0 (95% CI) | Average precision, split 0 (95% CI) |
|---|---|---|---|---|---|
| 1 | Age alone (rank, no model) | 0.795 (0.769 to 0.820) | 0.793 (0.770 to 0.810) | -0.011 (-0.017 to -0.005) | 0.046 (0.037 to 0.056) |
| 2 | Age + sex | 0.806 (0.783 to 0.829) | 0.799 (0.777 to 0.815) | reference | 0.051 (0.041 to 0.063) |
| 3 | Age + sex + six rhythm labels + normal_ecg | 0.821 (0.798 to 0.843) | 0.813 (0.797 to 0.823) | +0.015 (0.005 to 0.026) | 0.060 (0.048 to 0.080) |
| 4 | Age + sex + ECG-age gap | 0.811 (0.789 to 0.833) | 0.806 (0.780 to 0.820) | +0.005 (-0.003 to 0.013) | 0.059 (0.047 to 0.080) |
| 5 | Age + sex + Lima network features (ridge) | 0.840 (0.818 to 0.862) | 0.838 (0.826 to 0.846) | +0.035 (0.018 to 0.053) | 0.070 (0.057 to 0.090) |
| 5b | Lima network features alone (ridge, no age or sex) | 0.821 (0.799 to 0.845) | 0.822 (0.804 to 0.832) | +0.016 (-0.007 to 0.038) | 0.062 (0.050 to 0.081) |

Intervals are 95% bootstrap intervals over the test patients of split 0. The range across the five splits shows how much the choice of split moves each row: row 5 runs from 0.826 to 0.846.

## Published, on the authors' own test split (not ours)

Gu et al., Nat Mach Intell 2026, doi 10.1038/s42256-026-01180-5. These are their numbers on their split. Their split is not published, so no row above can be said to beat or match them; the two sets of numbers are side by side, not head to head.

| Source | Lead setting | From scratch | CSFM |
|---|---|---|---|
| Results text | not stated | 0.816 | up to 0.844 |
| Source Data Fig. 3a, best mean of 3 seeds | 12-Lead | 0.834 (ResNet1d18) | 0.862 (CSFM-Base) |
| Source Data Fig. 3a, best mean of 3 seeds | 6-Lead | 0.815 (Inception1d) | 0.841 (CSFM-Large) |
| Source Data Fig. 3a, best mean of 3 seeds | 2-Lead | 0.804 (ResNet1d34) | 0.841 (CSFM-Large) |
| Source Data Fig. 3a, best mean of 3 seeds | Lead II | 0.794 (ResNet1d18) | 0.819 (CSFM-Base) |

- The Lima network reads all 12 leads, so the 12-lead line is the like-for-like one. The text values 0.816 and 0.844 do not appear exactly in the Fig. 3a sheet; they are close to the six-lead values, and Fig. 2d is a six-lead result. That they are six-lead numbers is an inference (UNVERIFIED), and a fair thing to ask.
- No age, demographic or tabular baseline for this task was found in the Results text, the Fig. 3 caption or the Fig. 3a source data (TESTED for those three; the Methods were not read word for word).

## Calibration (split 0)

Slope: 1 is ideal, below 1 means predictions are too extreme. Intercept in the large: 0 is ideal.

| Model | Patients | Slope (95% CI) | Intercept in the large (95% CI) | Mean predicted | Observed |
|---|---|---|---|---|---|
| Age + sex | all test | 1.08 (0.96 to 1.22) | +0.01 (-0.12 to +0.12) | 1.24% | 1.25% |
| Age + sex + ECG-age gap | all test | 1.06 (0.95 to 1.19) | +0.01 (-0.11 to +0.11) | 1.24% | 1.25% |
| Age + sex + Lima network features (ridge) | all test | 0.99 (0.89 to 1.08) | +0.00 (-0.12 to +0.12) | 1.25% | 1.25% |
| Lima network features alone (ridge, no age or sex) | all test | 0.96 (0.87 to 1.06) | +0.01 (-0.12 to +0.12) | 1.24% | 1.25% |

## Decision curve (split 0, all test patients): net benefit per 100 patients

| Threshold | Treat all | Row 2, age + sex | Row 5, + Lima features | Row 5 minus row 2 (95% CI) | Share flagged by row 5 |
|---|---|---|---|---|---|
| 0.5% | 0.754 | 0.885 | 0.899 | +0.014 (-0.026 to +0.054) | 46.2% |
| 1.0% | 0.253 | 0.655 | 0.769 | +0.114 (+0.063 to +0.168) | 31.1% |
| 2.0% | -0.765 | 0.421 | 0.505 | +0.085 (+0.020 to +0.155) | 17.7% |
| 3.0% | -1.804 | 0.255 | 0.358 | +0.103 (+0.027 to +0.176) | 11.1% |
| 5.0% | -3.947 | 0.079 | 0.150 | +0.072 (-0.001 to +0.148) | 4.9% |
| 7.5% | -6.757 | 0.013 | 0.044 | +0.032 (-0.029 to +0.092) | 2.2% |
| 10.0% | -9.722 | -0.000 | 0.035 | +0.036 (-0.014 to +0.087) | 1.1% |

Treat none is 0 at every threshold. Net benefit as in Vickers and Elkin 2006, using `../decision-curves/netbenefit.py`.

## Subgroups (split 0, all test patients)

| Subgroup | Patients | Deaths | Row 2 | Row 4 | Row 5 | Row 5 minus row 2 (95% CI) |
|---|---|---|---|---|---|---|
| male | 9,094 | 163 | 0.766 | 0.777 | 0.809 | +0.042 (0.020 to 0.065) |
| female | 13,386 | 118 | 0.830 | 0.830 | 0.858 | +0.028 (0.002 to 0.056) |
| age 16 to 39 | 7,575 | 16 | 0.767 | 0.783 | 0.784 | +0.018 (-0.147 to 0.167) |
| age 40 to 59 | 6,568 | 36 | 0.674 | 0.701 | 0.778 | +0.104 (0.014 to 0.196) |
| age 60 to 74 | 4,898 | 74 | 0.596 | 0.605 | 0.696 | +0.100 (0.015 to 0.187) |
| age 75 and over | 3,439 | 155 | 0.614 | 0.636 | 0.694 | +0.080 (0.037 to 0.128) |

The youngest band has 16 deaths in the test split, so its numbers carry little weight. Within an age band, age and sex rank poorly by construction, which is why the within-band AUCs are lower than the overall ones.

## Sensitivity A: short follow-ups kept as survivors (split 0)

| Row | Main analysis | Sensitivity A |
|---|---|---|
| 1 Age alone (rank, no model) | 0.795 | 0.796 (0.771 to 0.820) |
| 2 Age + sex | 0.806 | 0.807 (0.785 to 0.829) |
| 3 Age + sex + six rhythm labels + normal_ecg | 0.821 | 0.822 (0.800 to 0.844) |
| 4 Age + sex + ECG-age gap | 0.811 | 0.812 (0.791 to 0.833) |
| 5 Age + sex + Lima network features (ridge) | 0.840 | 0.841 (0.819 to 0.862) |
| 5b Lima network features alone (ridge, no age or sex) | 0.821 | 0.819 (0.796 to 0.842) |

Sensitivity A test set: 23,364 patients, 281 deaths.

## Row 5 details

- 1,876 of 5,120 feature columns vary on the training patients of split 0; the other 3,244 are constant (dead units) and are dropped.
- Penalty chosen by validation log loss. Chosen C (scikit-learn, inverse penalty) per split: 0.003162, 0.003162, 0.003162, 0.003162, 0.003162.
- The network that produced the features was trained on the full CODE set with CODE-15% held out (Lima et al. 2021), so the features are out of sample for these patients.

## Checks (all must pass before a number is used)

1. Age alone on the whole main cohort: 0.7872; the pilot had 0.787. PASS.
2. AUC by the rank formula written out by hand against scikit-learn, every row and split: largest difference 2.2e-16. PASS.
3. Labels shuffled once, everything refitted: 1: 0.511, 2: 0.526, 3: 0.522, 4: 0.526, 5b: 0.502, 5: 0.502. PASS.
4. Patients in two splits: 0. PASS.
5. Row 4 against age + sex + predicted age: AUC difference 0.0e+00. PASS.
6. Stop rule: largest row 5 AUC over the five splits is 0.846, not above 0.86. PASS.
7. Every ridge fit converged and the chosen penalty is inside the grid. PASS.
8. Patients without outcome-exam features: 0; rows 1 to 4 give the same AUC on the whole test split and on the patients with features, every split. PASS.

## Limits

- Our splits are not the authors' split. Nothing here is a head-to-head comparison with CSFM.
- How the authors labelled patients followed for less than a year is not public (UNVERIFIED). The two readings tried here change the AUCs only in the third decimal.
- Row 5 is a linear model on a frozen network's last layer. It says what that network already encodes, not what a fine-tuned network could reach.
- One cohort, in Brazil. Death within one year is rare here (about 1.25%), so average precision is low for every row.
