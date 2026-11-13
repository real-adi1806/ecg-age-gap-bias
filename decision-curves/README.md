# Decision curves and proportional-hazards checks

`gap-over-age-sex/` asks whether the ECG-age gap adds to age and sex in predicting death on CODE-15%. This folder asks two follow-up questions about the same held-out models:

1. **Does the gap change decisions?** A decision curve compares net benefit (true positives per patient, minus false positives per patient weighted by the threshold odds) for the age and sex model with and without the gap.
2. **Do the Cox models' proportional-hazards assumptions hold well enough?** Every Cox model the analysis reports is tested, refitted within follow-up periods, and refitted with the baseline stratified by sex and age band.

**Not a medical device. Not for clinical use.**

## Results

Held-out 20% of patients (46,730 patients, 1,721 deaths). The thresholds are two-year risks of death from 0.25% to 20%. The summary band, 0.5% to 10%, was set in the script before any curve was computed. It is our choice, not a clinical standard.

| | 2 years | 4 years |
|---|---|---|
| Net benefit of adding the gap, mean over the band, per 1,000 patients (95% CI) | +0.28 (0.10 to 0.47) | +0.56 (0.26 to 0.84) |
| Band thresholds where the interval is above zero | 10 of 39 | 18 of 39 |

- Held-out concordance: 0.790 for age and sex, 0.799 with the gap.
- The gain is real but small: at 2 years it is about 8% of what age and sex already add over treating everyone or no one.
- No gap term failed the Grambsch and Therneau test (rank-time p 0.096 to 0.56). Sex did, in the five whole-cohort models (p 0.039 to 0.044).
- Stratifying the baseline by sex and age band moved the gap hazard ratios by at most 3.5%.

Every number is in `results_dca.json` and `results_ph.json`. The console output of the recorded runs is in `run_output_dca.txt` and `run_output_ph.txt`. Methods are in `METHODS.md`.

## Run it

From the repository root, with `data/exams.csv` downloaded as the main README describes:

```
pip install -r decision-curves/requirements.txt
python decision-curves/dca_paper_a.py        # about 2 minutes
make cox                                     # writes results/cox.json, which the next step reads
python decision-curves/ph_check_paper_a.py   # about 9 minutes
python decision-curves/merge_results.py
python decision-curves/make_figures_a.py
python -m pytest decision-curves/tests -q
```

`dca_paper_a.py` imports `gap-over-age-sex/analysis.py` unchanged and stops unless it reproduces that folder's held-out concordance (0.789967 and 0.798810). `ph_check_paper_a.py` stops unless lifelines reproduces the coefficients in `results/cox.json`.

## Files

| File | What it is |
|---|---|
| `dca_paper_a.py` | Decision curves at 2 and 4 years, calibration by decile, bootstrap intervals |
| `ph_check_paper_a.py` | Proportional-hazards tests, period-specific and stratified refits |
| `netbenefit.py` | Net benefit for binary and censored outcomes, tested by hand-worked examples in `tests/` |
| `make_figures_a.py` | The four figures in `figures/` |
| `merge_results.py` | Joins the two results files into `results.json` |
