# One-year mortality on CODE-15%: the rows under a foundation model

Generated from `results/results.json` by `make_summary.py` (1000 bootstrap resamples, seed base 20261114). Every number below was computed in this repository (TESTED) unless it sits under "Published", where it is copied from the paper or its source data.

## Cohort

- 233,647 patients with a death flag and a follow-up time in `exams.csv`, one row each.
- 2,812 died within one year; 222,057 were followed for at least one year; 8,778 were alive with under one year of follow-up.
- Main analysis: the 8,778 are left out. Death rate 1.25%. Keeping them as survivors (sensitivity A) gives 1.20%. The paper reports 1.23%, which sits between the two.
- Split by patient, 80/10/10, stratified on the label, five times (splits 0 to 4). Split 0: 179,868 train (2,250 deaths), 22,521 validation (281 deaths), 22,480 test (281 deaths).
- The saved network features belong to the outcome exam for 217,932 patients. For the other 15,715 the saved row is a different, usually later, exam, so they cannot be used for row 5 without leaking the future. Those patients all have two or more exams and a 1-year death rate of 0.20% against 1.33% in the rest, so the feature subset is not a random sample.

## The ladder

Table A. Test patients whose outcome exam has saved features (split 0: 20,892 patients, 278 deaths). All six rows are scored on the same patients, so the differences are paired.

| Row | Inputs | Test AUC, split 0 (95% CI) | Mean of 5 splits (range) | Difference from row 2, split 0 (95% CI) | Average precision, split 0 (95% CI) |
|---|---|---|---|---|---|
| 1 | Age alone (rank, no model) | 0.799 (0.775 to 0.823) | 0.795 (0.773 to 0.813) | -0.010 (-0.016 to -0.004) | 0.049 (0.040 to 0.060) |
| 2 | Age + sex | 0.809 (0.786 to 0.832) | 0.801 (0.780 to 0.817) | reference | 0.054 (0.044 to 0.068) |
| 3 | Age + sex + six rhythm labels + normal_ecg | 0.825 (0.803 to 0.848) | 0.817 (0.802 to 0.827) | +0.016 (0.006 to 0.026) | 0.064 (0.051 to 0.085) |
| 4 | Age + sex + ECG-age gap | 0.814 (0.791 to 0.837) | 0.808 (0.784 to 0.822) | +0.005 (-0.003 to 0.013) | 0.064 (0.051 to 0.088) |
| 5 | Age + sex + Lima network features (ridge) | 0.843 (0.821 to 0.865) | 0.840 (0.829 to 0.847) | +0.034 (0.017 to 0.052) | 0.077 (0.062 to 0.098) |
| 5b | Lima network features alone (ridge, no age or sex) | 0.824 (0.800 to 0.848) | 0.824 (0.808 to 0.833) | +0.015 (-0.007 to 0.037) | 0.068 (0.055 to 0.090) |

Table B. Every test patient (split 0: 22,480 patients, 281 deaths). Rows 1 to 4 only.

| Row | Inputs | Test AUC, split 0 (95% CI) | Mean of 5 splits (range) | Difference from row 2, split 0 (95% CI) | Average precision, split 0 (95% CI) |
|---|---|---|---|---|---|
| 1 | Age alone (rank, no model) | 0.795 (0.771 to 0.819) | 0.793 (0.770 to 0.810) | -0.011 (-0.017 to -0.005) | 0.046 (0.038 to 0.057) |
| 2 | Age + sex | 0.806 (0.784 to 0.827) | 0.799 (0.777 to 0.815) | reference | 0.051 (0.041 to 0.064) |
| 3 | Age + sex + six rhythm labels + normal_ecg | 0.821 (0.800 to 0.843) | 0.813 (0.797 to 0.823) | +0.015 (0.006 to 0.026) | 0.060 (0.048 to 0.081) |
| 4 | Age + sex + ECG-age gap | 0.811 (0.790 to 0.833) | 0.806 (0.780 to 0.820) | +0.005 (-0.003 to 0.013) | 0.059 (0.047 to 0.079) |

Intervals are 95% bootstrap intervals over the test patients of split 0. The range across the five splits shows how much the choice of split moves each row: row 5 runs from 0.829 to 0.847.

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
| Age + sex | feature subset | 1.08 (0.97 to 1.21) | +0.08 (-0.03 to +0.20) | 1.23% | 1.33% |
| Age + sex + ECG-age gap | feature subset | 1.07 (0.96 to 1.20) | +0.08 (-0.04 to +0.19) | 1.23% | 1.33% |
| Age + sex + Lima network features (ridge) | feature subset | 1.00 (0.91 to 1.09) | +0.01 (-0.11 to +0.13) | 1.32% | 1.33% |
| Lima network features alone (ridge, no age or sex) | feature subset | 0.97 (0.87 to 1.07) | +0.01 (-0.11 to +0.13) | 1.32% | 1.33% |
| Age + sex | all test | 1.08 (0.96 to 1.20) | +0.01 (-0.12 to +0.13) | 1.24% | 1.25% |
| Age + sex + ECG-age gap | all test | 1.06 (0.93 to 1.19) | +0.01 (-0.12 to +0.12) | 1.24% | 1.25% |

Rows 2 and 4 are fitted on every training patient, so on the feature subset (a higher-risk group) they under-predict a little; that is the positive intercept.

## Decision curve (split 0, feature subset): net benefit per 100 patients

| Threshold | Treat all | Row 2, age + sex | Row 5, + Lima features | Row 5 minus row 2 (95% CI) | Share flagged by row 5 |
|---|---|---|---|---|---|
| 0.5% | 0.835 | 0.962 | 0.974 | +0.012 (-0.031 to +0.051) | 47.3% |
| 1.0% | 0.334 | 0.730 | 0.842 | +0.112 (+0.059 to +0.169) | 32.0% |
| 2.0% | -0.683 | 0.489 | 0.566 | +0.077 (+0.003 to +0.156) | 18.7% |
| 3.0% | -1.721 | 0.300 | 0.415 | +0.116 (+0.038 to +0.193) | 12.1% |
| 5.0% | -3.862 | 0.096 | 0.204 | +0.107 (+0.029 to +0.189) | 5.5% |
| 7.5% | -6.670 | 0.015 | 0.070 | +0.055 (-0.023 to +0.124) | 2.5% |
| 10.0% | -9.633 | 0.000 | 0.039 | +0.039 (-0.018 to +0.096) | 1.2% |

Treat none is 0 at every threshold. Net benefit as in Vickers and Elkin 2006, using `../decision-curves/netbenefit.py`.

## Subgroups (split 0, feature subset)

| Subgroup | Patients | Deaths | Row 2 | Row 4 | Row 5 | Row 5 minus row 2 (95% CI) |
|---|---|---|---|---|---|---|
| male | 8,531 | 161 | 0.771 | 0.781 | 0.812 | +0.041 (0.019 to 0.066) |
| female | 12,361 | 117 | 0.834 | 0.833 | 0.862 | +0.029 (0.003 to 0.058) |
| age 16 to 39 | 7,234 | 16 | 0.767 | 0.782 | 0.784 | +0.017 (-0.140 to 0.174) |
| age 40 to 59 | 6,062 | 35 | 0.672 | 0.698 | 0.784 | +0.112 (0.020 to 0.205) |
| age 60 to 74 | 4,408 | 72 | 0.596 | 0.606 | 0.695 | +0.099 (0.015 to 0.187) |
| age 75 and over | 3,188 | 155 | 0.609 | 0.632 | 0.693 | +0.084 (0.039 to 0.130) |

The youngest band has 16 deaths in the test split, so its numbers carry little weight. Within an age band, age and sex rank poorly by construction, which is why the within-band AUCs are lower than the overall ones.

## Sensitivity A: short follow-ups kept as survivors (split 0)

| Row | Main analysis | Sensitivity A |
|---|---|---|
| 1 Age alone (rank, no model) | 0.799 | 0.800 (0.775 to 0.823) |
| 2 Age + sex | 0.809 | 0.810 (0.787 to 0.832) |
| 3 Age + sex + six rhythm labels + normal_ecg | 0.825 | 0.826 (0.805 to 0.847) |
| 4 Age + sex + ECG-age gap | 0.814 | 0.815 (0.793 to 0.835) |
| 5 Age + sex + Lima network features (ridge) | 0.843 | 0.844 (0.823 to 0.864) |
| 5b Lima network features alone (ridge, no age or sex) | 0.824 | 0.822 (0.800 to 0.844) |

Feature subset of the test split in both columns. Sensitivity A test set: 21,776 patients, 278 deaths.

## Row 5 details

- 1,874 of 5,120 feature columns vary on the training patients of split 0; the other 3,246 are constant (dead units) and are dropped.
- Penalty chosen by validation log loss. Chosen C (scikit-learn, inverse penalty) per split: 0.003162, 0.003162, 0.003162, 0.003162, 0.003162.
- The network that produced the features was trained on the full CODE set with CODE-15% held out (Lima et al. 2021), so the features are out of sample for these patients.

## Checks (all must pass before a number is used)

1. Age alone on the whole main cohort: 0.7872; the pilot had 0.787. PASS.
2. AUC by the rank formula written out by hand against scikit-learn, every row and split: largest difference 2.2e-16. PASS.
3. Labels shuffled once, everything refitted: 1: 0.517, 2: 0.533, 3: 0.527, 4: 0.531, 5b: 0.504, 5: 0.505. PASS.
4. Patients in two splits: 0. PASS.
5. Row 4 against age + sex + predicted age: AUC difference 0.0e+00. PASS.
6. Stop rule: largest row 5 AUC over the five splits is 0.847, not above 0.86. PASS.
7. Every ridge fit converged and the chosen penalty is inside the grid. PASS.

## Limits

- Our splits are not the authors' split. Nothing here is a head-to-head comparison with CSFM.
- How the authors labelled patients followed for less than a year is not public (UNVERIFIED). The two readings tried here change the AUCs only in the third decimal.
- Row 5 uses 209,154 of the 224,869 main-cohort patients; the left-out group holds 32 of the 2,812 deaths.
- Row 5 is a linear model on a frozen network's last layer. It says what that network already encodes, not what a fine-tuned network could reach.
- One cohort, in Brazil. Death within one year is rare here (about 1.25%), so average precision is low for every row.
