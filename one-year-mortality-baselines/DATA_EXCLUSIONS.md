# Data exclusions: who is left out of the CODE-15% ladder, and why

Written by `exclusions.py` from `exams.csv`, the feature id files, `results/results.json` (this version), `results/results-v1.json` (the first published version, commit b952482) and `results/outcome_extraction.json`. Every number below is in `results/exclusions.json`; none is typed by hand. The script stops without writing if a count differs from those files.

**In this version the only patients left out are the 8,778 whose one-year outcome is unknown.** The first version also left out 15,715 patients whose saved network features came from a different exam than the one the outcome is attached to. Their outcome exams have since been run through the network, and they are back in every row. Row 5 moved from 0.843 to 0.840 on split 0 (section C).

## A. The cohort, step by step

```
345,779 exams in exams.csv
233,770 patients
  - 123 patients with no exam that carries death and timey (190 exams)
= 233,647 patients with an outcome, 2,812 deaths within 1 year (1.20%)
  - 8,778 alive with under 1 year of follow-up (0 deaths by definition)
= 224,869 main cohort, 2,812 deaths (1.25%), all with features from the outcome exam
  split 0 (seed 20261114): train 179,868 (2,250 deaths), validation 22,521 (281), test 22,480 (281)
```

Each of the 233,647 patients has exactly one exam with death and timey (the most on any patient is 1). Network features for that exam: 217,932 patients from the saved features of each patient's lowest exam id, where that exam is the outcome exam, and 15,715 extracted from the outcome exam's trace by `extract_outcome_exams.py`. The extracted predicted ages match `nn_predicted_age` in `exams.csv`, the network's prediction that ships with the dataset, with a largest difference of 0.0005 years (Pearson r 1.000000), so the right traces were read. Missing values: none in any column used here, so no patient is left out for a missing value.

The first version had one more step:

```
  224,869 main cohort
  - 15,715 saved features not from the outcome exam (32 deaths, 0.20%)
  = 209,154 kept, 2,780 deaths (1.33%); split 0 test 20,892 (278 deaths)
```

The short-follow-up patients and these 15,715 do not overlap (0 in both). The split is drawn over all 233,647 patients with the same seed in both versions, so the first version's 20,892 test patients are a subset of this version's 22,480.

## B. The patients the first version left out, and the exact logic

CODE-15% has no exam dates or timestamps, so nothing below is date logic. The saved features come from `germany/extract_code.py`, which keeps one exam per patient, the lowest exam id:

```
first = df.sort_values("exam_id").drop_duplicates("patient_id")  # first exam per patient
```

The outcome (death, timey) sits on one exam per patient, and for 217,932 of 233,647 patients that is the lowest exam id. For the other 15,715, `ladder.load_features`, which matches on exam id, found no saved features for the outcome exam, and the patient dropped out of every row. All of them have two or more exams (fewest: 2).

What the sources say about which exam carries the outcome:

Zenodo record 4916206, description of `exams.csv` (read through the Zenodo API):

> "death": true if the patient dies in the follow-up time. This data is available only in the first exam of the patient. Other exams will have this as an empty field

Lima et al. 2021, Nature Communications 12:5117, article text on the journal page:

> Only the first patient exam is considered in all the analysis with this cohort and the remaining exams from the patients are removed from the remaining data and not used in the analysis.

Neither defines "first" (by date, by id, or otherwise), and the sources read do not say whether exam id follows time. So the order has to be judged from the data. Age is in whole years, so it gives a direction only when the years differ:

| Saved-feature exam compared with the outcome exam | Patients | Share of 15,715 |
|---|---|---|
| Older (higher age in years) | 14,734 | 93.8% |
| Same age in years | 904 | 5.8% |
| Younger | 77 | 0.5% |
| Lower exam id | 15,715 | 100.0% |
| Higher exam id | 0 | 0.0% |

The id rows are true by construction: the saved exam is the lowest id, so it can never be higher. The median age difference is 3 years. In 15,563 of the 15,715 patients (99.0%) the outcome exam has the lowest age among the patient's exams, against 99.3% of the 50,223 kept patients with two or more exams. Exams.csv defines `age` as "patient age in years at the moment of the exam" (same Zenodo description). Read that way, the older group is a later exam, the same-age group cannot be ordered (904 patients), and the younger group (77 patients) is an earlier exam. Exam id order is not documented as chronological, and nothing here shows that it is.

Why that mattered: an exam recorded after the outcome exam exists only if the patient was alive at that later date, so features taken from it carry information about survival after the outcome exam, which is what a one-year death label asks the model to predict. The first version removed these patients rather than estimate how large that effect would be. This version reads the outcome exam itself, so the question no longer arises.

## C. Why leaving them out was leak prevention, and what changed when they came back

The patients left out die less: 32 deaths in 15,715 (0.20%) against 2,780 in 209,154 (1.33%). Removing a low-risk group could in principle move an AUC either way, so the test is to put them back and look. Split 0, AUC for death within one year with 95% bootstrap intervals:

| Row | First version, 20,892 test patients (278 deaths) | This version, 22,480 (281 deaths) | Change | Mean of 5 splits, first version to this one |
|---|---|---|---|---|
| 1. Age alone | 0.799 (0.775 to 0.823) | 0.795 (0.769 to 0.820) | -0.004 | 0.795 to 0.793 |
| 2. Age + sex | 0.809 (0.786 to 0.832) | 0.806 (0.783 to 0.829) | -0.003 | 0.801 to 0.799 |
| 3. Age + sex + rhythm labels | 0.825 (0.803 to 0.848) | 0.821 (0.798 to 0.843) | -0.004 | 0.817 to 0.813 |
| 4. Age + sex + ECG-age gap | 0.814 (0.791 to 0.837) | 0.811 (0.789 to 0.833) | -0.003 | 0.808 to 0.806 |
| 5. Age + sex + Lima features | 0.843 (0.821 to 0.865) | 0.840 (0.818 to 0.862) | -0.003 | 0.840 to 0.838 |
| 5b. Lima features alone | 0.824 (0.800 to 0.848) | 0.821 (0.799 to 0.845) | -0.003 | 0.824 to 0.822 |

Row 5 minus row 2 on split 0: +0.034 (0.017 to 0.052) in the first version, +0.035 (0.018 to 0.053) in this one. The two columns are different patient sets, so the change has no interval. Rows 1 to 4 use no network features: on the whole test split they are identical in the two versions (largest difference exactly 0, checked by this script), so their change above comes only from scoring the added patients.

Limits:

- The added patients hold 32 of the 2,812 deaths and 3 of the 281 in split 0's test set, so they move a pooled AUC only a little. The small change shows the exclusion did not flatter the results. It does not show how well any row ranks risk within that group.
- The extracted features come from the same network, read the same way, as the saved ones; the prediction check in section A is the evidence for that.

## D. Baseline characteristics

| | With an outcome | Main cohort | Left out of the first version | Kept in the first version | Split 0 test, this version | SMD, left out vs kept |
|---|---|---|---|---|---|---|
| Patients | 233,647 | 224,869 | 15,715 | 209,154 | 22,480 |  |
| Deaths within 1 year | 2,812 (1.20%) | 2,812 (1.25%) | 32 (0.20%) | 2,780 (1.33%) | 281 (1.25%) | -0.13 |
| Age, mean (SD) | 50.7 (19.8) | 50.8 (19.8) | 55.5 (17.6) | 50.5 (19.9) | 50.7 (19.8) | +0.27 |
| Age, median (Q1 to Q3) | 51 (34 to 68) | 51 (34 to 68) | 57 (42 to 70) | 50 (33 to 68) | 51 (34 to 68) |  |
| Male, % | 40.5 | 40.6 | 37.6 | 40.8 | 40.5 | -0.07 |
| Follow-up, years, median (Q1 to Q3) | 3.46 (2.10 to 5.21) | 3.56 (2.25 to 5.30) | 6.13 (5.70 to 6.76) | 3.38 (2.17 to 4.73) | 3.56 (2.25 to 5.26) | +1.76 |
| Exams per patient, median (Q1 to Q3) | 1 (1 to 2) | 1 (1 to 2) | 3 (2 to 4) | 1 (1 to 1) | 1 (1 to 2) |  |
| Two or more exams, % | 28.6 | 29.3 | 100.0 | 24.0 | 29.9 | +2.52 |
| 1st degree AV block, % | 1.4 | 1.4 | 1.8 | 1.4 | 1.4 | +0.03 |
| Right bundle branch block, % | 2.4 | 2.5 | 4.0 | 2.3 | 2.4 | +0.10 |
| Left bundle branch block, % | 1.4 | 1.5 | 1.9 | 1.4 | 1.5 | +0.04 |
| Sinus bradycardia, % | 1.6 | 1.6 | 2.0 | 1.6 | 1.6 | +0.03 |
| Sinus tachycardia, % | 2.2 | 2.2 | 1.8 | 2.2 | 2.3 | -0.03 |
| Atrial fibrillation, % | 1.5 | 1.5 | 1.8 | 1.5 | 1.5 | +0.02 |
| Normal ECG label, % | 39.4 | 38.7 | 0.1 | 41.6 | 38.9 | -1.19 |
| ECG-age gap, years, mean (SD) | 1.2 (11.0) | 1.2 (11.0) | 0.3 (10.9) | 1.2 (11.0) | 1.2 (10.9) | -0.08 |

All characteristics are taken from the outcome exam. The gap is `nn_predicted_age` minus `age`. SMD is the left-out mean minus the kept mean over the pooled SD (for percentages, the same on the 0 or 1 values); for follow-up it uses means, not the medians shown. Follow-up is time to death for the dead and time to last contact for the rest. Exams per patient counts every exam of the patient in `exams.csv`, which is why the left-out group all have two or more.

Normal ECG label. Take the 66,894 patients with an outcome and two or more exams (this counts the short follow-up patients too, so the lowest-id group below has 956 more than the kept count in section B). The outcome exam carries `normal_ecg` in 0.1% (15,715 patients) when it is not the patient's lowest exam id, and in 39.6% (51,179 patients) when it is. The 112,132 exams that do not carry the outcome have it in 37.9%. On the outcome exam, 87.6% of the left-out patients have neither `normal_ecg` nor any of the six rhythm labels, against 49.1% of the kept patients with two or more exams and 48.7% of the kept patients with one exam. The Zenodo description defines the label as: "normal_ecg": True if automatic annotation system say it is a normal ECG. We do not know why the outcome exam of these patients almost never carries it, and the dataset description does not say how exam ids were assigned. Row 3 uses `normal_ecg`, so it is the row most exposed to this: on split 0 it moved from 0.825 to 0.821 when these patients came back (section C).

## Reproduce

```
cd one-year-mortality-baselines
python extract_outcome_exams.py    # once: writes the code_outcome_part files, needs the CODE-15% HDF5 traces
python analysis.py --boot 1000
python exclusions.py               # needs ../data/exams.csv and the _ids.npy files; reads no _feat.npy
```
