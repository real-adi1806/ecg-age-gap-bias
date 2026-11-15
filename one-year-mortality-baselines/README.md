# What sits under a foundation model's 1-year mortality AUC on CODE-15%?

The parent folder replicates the age-bias correction of the AI ECG age gap on the Brazilian CODE-15% cohort, and `../gap-over-age-sex` asks what the gap adds to age and sex. This folder asks the same kind of question about a newer result.

Gu et al. (Nature Machine Intelligence, 2026) pretrained a cardiac foundation model, CSFM, and report its AUC for death within one year of the ECG on CODE-15%, against neural networks trained from scratch. The comparison is between networks. This folder adds the rows underneath: age alone, age and sex, the rhythm labels that ship with the dataset, the ECG-age gap, and a linear model on the frozen features of the 2021 ECG-age network of Lima et al.

**Not a medical device. Not for clinical use.**

**The splits here are not the authors' split, which is not published. No row below can be said to beat or match CSFM; the published numbers are shown beside ours, not against them.**

## Cohort

```
233,647 patients in CODE-15% exams.csv with a death flag and follow-up time
        (the count the paper's supplement gives for CODE-15)
  - 8,778   alive at last contact with under 1 year of follow-up: outcome unknown
            (kept as survivors in sensitivity A; every row moves by 0.002 or less)
= 224,869   main cohort, 2,812 deaths within 1 year (1.25%)
  split 0 test set: 22,480 patients, 281 deaths
```

That is the only exclusion. Every patient's network features come from the exam the outcome is attached to. The first published version left out 15,715 more patients, because the saved features came from a different exam; their outcome exams have since been extracted and they are back in every row. [`DATA_EXCLUSIONS.md`](DATA_EXCLUSIONS.md) gives the cohort step by step, the exact logic, and what changed (every row moved by 0.004 or less; row 5 from 0.843 to 0.840).

## Results

Held-out patients of split 0: 22,480 patients, 281 deaths. AUC for death within one year, with 95% bootstrap intervals (1,000 resamples of patients), and the mean and range over five different splits.

| Row | Inputs | Test AUC, split 0 (95% CI) | Mean of 5 splits (range) | Difference from row 2 (95% CI) |
|---|---|---|---|---|
| 1 | Age alone, as a rank | 0.795 (0.769 to 0.820) | 0.793 (0.770 to 0.810) | -0.011 (-0.017 to -0.005) |
| 2 | Age + sex | 0.806 (0.783 to 0.829) | 0.799 (0.777 to 0.815) | reference |
| 3 | Age + sex + six rhythm labels + normal_ecg | 0.821 (0.798 to 0.843) | 0.813 (0.797 to 0.823) | +0.015 (0.005 to 0.026) |
| 4 | Age + sex + ECG-age gap | 0.811 (0.789 to 0.833) | 0.806 (0.780 to 0.820) | +0.005 (-0.003 to 0.013) |
| 5 | Age + sex + frozen Lima network features (ridge) | 0.840 (0.818 to 0.862) | 0.838 (0.826 to 0.846) | +0.035 (0.018 to 0.053) |
| 5b | Frozen Lima network features alone (ridge) | 0.821 (0.799 to 0.845) | 0.822 (0.804 to 0.832) | +0.016 (-0.007 to 0.038) |

Published by Gu et al. on their own test split, for reference only:

| Source | From scratch | CSFM |
|---|---|---|
| Results text, lead setting not stated | 0.816 | up to 0.844 |
| Source Data for Fig. 3a, 12 leads, best mean of 3 seeds | 0.834 | 0.862 |
| Source Data for Fig. 3a, 6 leads, best mean of 3 seeds | 0.815 | 0.841 |

**Within age bands.** Most of the pooled AUC is age: older people die more. Inside each band that advantage is mostly gone, so the bands show whether the network carries anything beyond age (split 0, same test patients):

| Age band | Patients | Deaths | Age + sex (row 2) | + frozen features (row 5) | Row 5 minus row 2 (95% CI) |
|---|---|---|---|---|---|
| 16 to 39 | 7,575 | 16 | 0.767 | 0.784 | +0.018 (-0.147 to 0.167) |
| 40 to 59 | 6,568 | 36 | 0.674 | 0.778 | +0.104 (0.014 to 0.196) |
| 60 to 74 | 4,898 | 74 | 0.596 | 0.696 | +0.100 (0.015 to 0.187) |
| 75 and over | 3,439 | 155 | 0.614 | 0.694 | +0.080 (0.037 to 0.128) |

From 40 up, age and sex reach only 0.60 to 0.67 within a band, and the frozen network adds 0.08 to 0.10 with intervals above zero. The 16 to 39 band has 16 deaths and says little. So the ECG carries mortality signal beyond age; the question this folder asks is how much of a foundation model's headline number that older signal, plus age and sex, already accounts for.

- Age and sex alone reach 0.806 on split 0. A strong demographic baseline on CODE-15% is already known: Lukyanenko et al. report a concordance of 0.79 for death from demographics alone [4], and Lima et al. report an AUC of 0.80 (95% CI 0.79 to 0.81) for 1-year mortality from their Cox model of ECG-age groups adjusted for age and sex [2]. The rhythm labels add +0.015; the ECG-age gap adds +0.005, with an interval that includes zero.
- A linear model on the 2021 network's frozen features, with age and sex, reaches 0.840 on split 0 and between 0.826 and 0.846 across the five splits. The Lima network reads 12 leads, so the 12-lead published values are the like-for-like ones, and they were measured on a different split.
- That model is well calibrated on the held-out patients: slope 0.99 (95% CI 0.89 to 1.08), intercept in the large +0.00.
- On a decision curve it has higher net benefit than age and sex at every threshold from 0.5% to 10%. The 95% interval of the gain is above zero from 1% to 3.5%; from 4% to 6% it is above zero at some thresholds and touches zero at others. The gain is small in absolute terms because death within a year is rare (about 1.25%).
- Treating the 8,778 patients with under a year of follow-up as survivors instead of leaving them out moves every row by 0.002 or less.

The full tables, calibration, decision curve, subgroups, sensitivity analysis and checks are in [`results/SUMMARY.md`](results/SUMMARY.md). The figure is [`results/ladder.png`](results/ladder.png).

![The ladder, calibration and decision curve](results/ladder.png)

## What is true by construction, and is not a finding

Row 4 uses the ECG-age gap (predicted age minus age). With age already in the model, the gap and the predicted age carry the same information, so "age + sex + gap" and "age + sex + predicted age" give the same AUC. The run checks this and it holds exactly.

## How the cohort and label were built

- `exams.csv` has 233,647 patients with a death flag and a follow-up time, the same count as the paper's supplement gives for CODE-15.
- Label: died within one year of the exam. 2,812 patients.
- 8,778 patients were alive at last contact with under one year of follow-up, so their one-year outcome is unknown. The main analysis leaves them out (death rate 1.25%). Sensitivity A keeps them as survivors (1.20%). The paper reports 1.23%; how it treated these patients is not stated in what we could read.
- Five patient-level 80/10/10 splits, stratified on the label. Models are fitted on train, the ridge penalty is chosen on validation, and the test patients are scored once.

## Network features from the outcome exam

Row 5 needs the Lima network's features for the exam the outcome is attached to. The features saved earlier by `../germany/extract_code.py` hold one exam per patient, the lowest exam id, and for 15,715 of the 233,647 patients that is a different exam, in 94% of those cases a later one (the patient is older). Using it would leak the future. The first published version therefore left those patients out. `extract_outcome_exams.py` now runs the network on their outcome exams, and its predicted ages match `nn_predicted_age` in `exams.csv` to within 0.0005 years, so every row scores every patient. [`DATA_EXCLUSIONS.md`](DATA_EXCLUSIONS.md) has the full account, with a baseline table of the patients who came back.

## Other limits

- Row 5 is a linear model on a frozen network's last layer. It shows what that network already encodes; it is not a fine-tuned network.
- The Lima network reads all 12 leads. The paper's headline values in the text (0.844 and 0.816) do not appear in its Fig. 3a source data, where the 12-lead values are higher and the six-lead values are close to the text. Which lead setting the text refers to is our inference, not a stated fact.
- CODE-15% was held out from the training of the Lima network, so its features are out of sample here. Lima et al. used "the 85% split ... to develop the model", and it "is evaluated in three different cohorts, unseen by the DNN model during its development, the 15% hold-out split described above, which will be referred to as the CODE-15% cohort" [2].
- One cohort, in Brazil. About 280 deaths in each test split.

## Reproduce

Needs `exams.csv` from CODE-15% (Zenodo record 4916206) in `../data/`, and the network features: `code_part{0..17}_feat.npy` and `_ids.npy` (about 5 GB, produced by `../germany/extract_code.py` from the CODE-15% traces), plus the `code_outcome_part{0..17}` files that `extract_outcome_exams.py` writes beside them. Peak memory is about 10 GB.

```
cd one-year-mortality-baselines
python -m pip install -r requirements.txt
python tests/test_ladder.py        # 13 tests, a few seconds; also runs under pytest
python tests/test_exclusions.py    # 7 tests
python extract_outcome_exams.py    # once: needs the CODE-15% HDF5 traces and the network weights, under a minute on an Apple M5
python analysis.py --boot 1000     # writes results/results.json, about 10 minutes on an Apple M5
python make_summary.py             # writes results/SUMMARY.md from results.json
python make_figure.py              # writes results/ladder.png
python exclusions.py               # writes DATA_EXCLUSIONS.md and results/exclusions.json
```

If the files are elsewhere: `python analysis.py --data /path/to/exams.csv --features /path/to/features`; `extract_outcome_exams.py` also takes `--code-dir` and `--model-dir`.

The seeds are fixed, so the point estimates should match `results/results.json`. Every number in `SUMMARY.md` is written by `make_summary.py`; none is typed by hand. `make_summary.py` and `make_figure.py` refuse to run if any check in `results.json` failed. The decision curve reuses `../decision-curves/netbenefit.py`.

## Files

| File | What it is |
|---|---|
| `analysis.py` | The whole analysis |
| `extract_outcome_exams.py` | Network features for the outcome exams the saved files lack |
| `exclusions.py` | Writes `DATA_EXCLUSIONS.md`: the cohort flow and the patients the first version left out |
| `ladder.py` | Cohort, splits, AUC by hand, Newton logistic regression, calibration, bootstrap, feature loading |
| `tests/test_ladder.py`, `tests/test_exclusions.py` | Hand-worked tests of `ladder.py` and `exclusions.py` |
| `make_summary.py`, `make_figure.py` | Build `results/SUMMARY.md` and `results/ladder.png` from `results/results.json` |
| `results/published_fig3a.json` | The published AUCs for this task, read from the paper's Source Data for Fig. 3a |
| `results/run_output.txt` | The console output of the reported run |
| `results/outcome_extraction.json` | What `extract_outcome_exams.py` extracted, and its agreement with `nn_predicted_age` |
| `results/results-v1.json` | The first published version's results (15,715 patients left out), kept for the comparison in `DATA_EXCLUSIONS.md` |

## Data

CODE-15%: Ribeiro, A. H., Paixao, G. M. M., Lima, E. M., Horta Ribeiro, M. et al. *CODE-15%: a large scale annotated dataset of 12-lead ECGs*. Zenodo, version 1.0.0. https://doi.org/10.5281/zenodo.4916206. Creative Commons Attribution 4.0 International.

No patient data are stored in this repository.

## References

1. Gu, X., Tang, W., Han, J. et al. Cardiac health assessment across scenarios and devices using a multimodal foundation model pretrained on data from 1.7 million individuals. *Nature Machine Intelligence* 8(2), 220-233 (2026). https://doi.org/10.1038/s42256-026-01180-5
2. Lima, E. M., Ribeiro, A. H., Paixao, G. M. M. et al. Deep neural network-estimated electrocardiographic age as a mortality predictor. *Nature Communications* 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7
3. Vickers, A. J., Elkin, E. B. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making* 26(6), 565-574 (2006). https://doi.org/10.1177/0272989X06295361
4. Lukyanenko, P., Mayourian, J., Liu, M., Triedman, J. K., Ghelani, S. J., La Cava, W. G. Deep survival analysis from adult and pediatric electrocardiograms: a multi-center benchmark study. *BioData Mining* (2025). https://doi.org/10.1186/s13040-025-00510-4

## Author and licence

Adithya Krishnan, independent researcher.

Licence: same as the repository (see `../LICENSE`).
