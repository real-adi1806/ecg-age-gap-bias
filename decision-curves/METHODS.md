# Paper A: decision curve, horizon calibration and proportional hazards, methods

Written 12 Nov 2026. Every number below comes from `dca_paper_a.py` (`results_dca.json`,
console copy `run_output_dca.txt`) or `ph_check_paper_a.py` (`results_ph.json`,
`run_output_ph.txt`). Nothing is typed by hand.

## Verdict

1. The ECG-age gap adds net benefit over age and sex, and the gain is small. At a 2-year
   horizon, averaged over thresholds of 0.5% to 10%, it adds 0.28 net deaths identified
   per 1,000 patients (95% bootstrap CI 0.10 to 0.47). The age and sex model alone is
   already 3.35 per 1,000 ahead of the better default strategy, so the gap adds about 8%
   to that. TESTED.
2. Both models are well calibrated at 2 and 4 years (observed over expected 1.05 and 1.04);
   adding the gap does not change calibration. TESTED.
3. Proportional hazards hold for every gap term in every Cox model the manuscript reports
   (Grambsch and Therneau p 0.10 to 0.56, hazard ratios stable across follow-up periods). The
   only detectable departure is sex (p about 0.04; HR about 1.65 for 4 years, 1.39 after), and
   stratifying on sex and age band moves the gap hazard ratios by at most 3.5%. No conclusion
   changes. TESTED.

## Inputs, and proof they are the paper's

- Code imported read-only: `ecg-age-bias/gap-over-age-sex/analysis.py` (loader, split,
  correction, Cox fits) and, for the Cox models in the manuscript, `ecg-age-bias/cox_model.py`
  and `ecg_age_bias.py`. Nothing in that repository was changed; bytecode writing was off.
- Data: `ecg-age-bias/data/exams.csv` (md5 recorded in `results_dca.json`), Brazil only.
  233,647 patients, 8,341 deaths. Germany (PTB-XL) has no outcomes and is not used.
- Split: rebuilt exactly as in `analysis.py` (first permutation of
  `numpy.random.default_rng(20261108)`, 80/20 by patient): 186,917 training patients (6,620
  deaths) and 46,730 held-out patients (1,721 deaths). The script stops unless it reproduces
  the published held-out Harrell's C to 1e-9: 0.789967 (age and sex) and 0.798810 (age, sex
  and gap), and the gap hazard ratio 1.028841 per year.
- Raw and corrected gap: the predicted risks of the two gap models differ by at most 7e-16
  (2 years) and 1e-15 (4 years), so one curve covers both. This is an identity (the
  corrected gap is the raw gap minus a line in age), checked, not a finding.

## Predicted risk at the horizon

Each Cox model (lifelines, Efron ties, fitted on the training patients) gives
risk(tau | x) = 1 - S0(tau)^exp(x'beta), with S0 the Breslow baseline survival at tau.

## Horizon, fixed before any curve was computed

- Follow-up of all 233,647 patients: median 3.46 years; still under observation past
  1, 2, 3, 4 and 5 years: 95.0%, 77.2%, 58.9%, 40.5% and 27.2%.
- Primary horizon 2 years: 77.2% of patients are followed past it, so the Kaplan-Meier risk
  in each threshold group rests mostly on observed follow-up, and it is the shorter of the
  two Kaplan-Meier horizons the manuscript already uses. On the held-out patients 20.8% are
  censored before 2 years and 972 die by 2 years (Kaplan-Meier risk 2.21%).
- Sensitivity horizon 4 years, the manuscript's other horizon: 56.3% of held-out patients
  are censored before it, 1,457 die by it (Kaplan-Meier risk 3.91%). Reported in full, not
  only where it helps.

## Net benefit with censoring (Vickers et al. 2008; McLernon et al. 2023)

Patients with predicted risk >= threshold t are "treated". With n_pos of the n held-out
patients above t, and S_pos the Kaplan-Meier survival at tau among those n_pos only:

    TP = (1 - S_pos) x n_pos        FP = S_pos x n_pos
    NB(t) = TP/n - FP/n x t/(1 - t)
    treat all:  (1 - S) - S x t/(1 - t), S = Kaplan-Meier survival at tau of everyone
    treat none: 0

Treat all crosses zero exactly at the horizon risk 1 - S (checked to 1e-12 at both
horizons). Kaplan-Meier is the shared implementation in `shared/netbenefit.py`, which
matches lifelines to 1e-12 with tied times and censoring in the unit tests.

**Thresholds.** 0.25% to 20% in steps of 0.25% (80 points), plotted in full. The summary
band, 0.5% to 10%, was written into the script before the first run, chosen to bracket a
death rate of a few per cent from well below to several times above. The run then showed
it fits: the 2-year death risk is 2.2%, the middle 98% of predicted 2-year risks runs from
0.1% to 11.2%, and above 10% fewer than 2% of patients are flagged by either model. The
manuscript names no clinical use for a death-risk score, so no narrower band is claimed.

**Uncertainty.** 1,000 paired bootstrap resamples of the held-out patients (models held
fixed, as in `analysis.py`'s own bootstrap; seed 20261112 plus the horizon), percentile 95%
intervals for the difference in net benefit at each threshold and for its band and
full-range means.

**Cross-check on the real data.** Among held-out patients with no censoring before the
horizon (dead by it, or followed past it), the survival net benefit equals the binary net
benefit with outcome "died by tau" at every threshold, to 1e-12, at both horizons.

## Results: decision curve (TESTED)

| | 2 years (primary) | 4 years (sensitivity) |
|---|---|---|
| Kaplan-Meier risk, held-out | 2.21% | 3.91% |
| gap model minus age and sex, mean over 0.5% to 10% | +0.00028 (0.00010 to 0.00047) | +0.00056 (0.00026 to 0.00084) |
| same, per 1,000 patients | +0.28 (0.10 to 0.47) | +0.56 (0.26 to 0.84) |
| same, mean over 0.25% to 20% | +0.00018 (0.00008 to 0.00029) | +0.00053 (0.00027 to 0.00079) |
| thresholds where the 95% CI is above zero | 14 of 80 | 31 of 80 |
| thresholds where it is below zero | 0 | 0 |
| age and sex minus better default, band mean | +0.00335 (0.00279 to 0.00393) | +0.00930 (0.00835 to 0.01034) |
| gap model minus better default, band mean | +0.00363 (0.00302 to 0.00425) | +0.00986 (0.00892 to 0.01087) |

At single thresholds (2 years), per 1,000 patients: 0.5%, +0.26 (0.09 to 0.43); 1%, +0.25
(0.06 to 0.46); 2%, +0.21 (-0.14 to 0.55); 5%, +0.35 (-0.16 to 0.91); 10%, +0.10 (-0.29 to
0.48). Against the better of treat-all and treat-none, the 95% interval is above zero at
38 of the 39 band thresholds for the gap model at both horizons, and for age and sex at 31
of 39 at 2 years (not from 8.25% to 10%, where few patients are flagged) and 38 of 39 at
4 years. The gain from the gap is consistent in sign but its interval includes zero at
most single thresholds; it is the average over the band that is clearly above zero.

## Results: calibration at the horizon (TESTED)

Observed (Kaplan-Meier) over mean predicted risk on the held-out patients: 1.047 (age and
sex) and 1.046 (gap model) at 2 years; 1.037 and 1.037 at 4 years. Both models predict
slightly less death than occurs. By decile of predicted risk (Figure A3), the 2-year mean
predicted risk runs from 0.13% to 8.04% (age and sex) against observed 0.21% to 8.18%, and
the decile intervals mostly cover the diagonal; the gap model behaves the same way (0.11%
to 8.35% predicted, 0.15% to 8.51% observed). Decile intervals are Greenwood intervals on
the log(-log) scale from lifelines.

## Proportional hazards

### Models tested

All five Cox models the manuscript reports, on the full Brazilian cohort (233,647 patients,
8,341 deaths), each adjusted for age (per year) and sex:

| model | gap terms |
|---|---|
| raw_categorical | raw gap > +8 and < -8, against within +/-8 |
| corrected_categorical | the same on the corrected gap |
| corrected_categorical_share_matched | corrected gap cut at the raw cut's shares (> +6.620, < -8.582) |
| raw_continuous | raw gap per year |
| corrected_continuous | corrected gap per year |

plus the two held-out-analysis models fitted on the 186,917 training patients (age and sex;
age, sex and gap). The manuscript fitted its models with its own standard-library Efron code;
lifelines (Efron) reproduces every coefficient to 1.2e-6, and the script stops otherwise.

### Methods

1. Grambsch and Therneau test on scaled Schoenfeld residuals, rank and Kaplan-Meier time
   transforms (lifelines `proportional_hazard_test`).
2. Size of departure, two ways, because with 8,341 deaths a test could flag a trivial drift:
   - beta(t) = beta + scaled Schoenfeld residual, averaged in ten equal-count bins of death
     time with 95% intervals, and the least-squares slope of beta(t) on time in years
     (change in log hazard ratio per year of follow-up);
   - the model refitted separately within follow-up periods 0 to 1, 1 to 2, 2 to 4 and 4+
     years (left-truncated: everyone alive and in follow-up at the period start). The first
     period includes the 47 patients whose death is recorded at time 0; a first run dropped
     them, a test caught it (period deaths did not sum to 8,341) and the run was repeated.
3. Sensitivity: each model refitted stratified on 10-year age band (under 30 pooled) and sex,
   with linear age kept within strata. This frees the baseline hazard to have a different
   shape for each age band and sex, so a non-proportional age or sex effect cannot leak into
   the gap terms.

### Results (TESTED)

**The tests did not reject for any gap term, and the departures are small.** The brief
expected trivial rejection at this sample size; it did not happen.

| term (model) | HR | GT test p (rank) | change in log HR per year (95% CI) | HR by period 0-1, 1-2, 2-4, 4+ years | stratified HR |
|---|---|---|---|---|---|
| raw > +8 vs within | 1.774 | 0.50 | -0.006 (-0.041 to 0.028) | 1.793, 1.846, 1.647, 1.891 | 1.818 |
| raw < -8 vs within | 0.791 | 0.50 | +0.004 (-0.029 to 0.036) | 0.840, 0.767, 0.712, 0.888 | 0.783 |
| corrected > +8 vs within | 1.586 | 0.10 | -0.022 (-0.052 to 0.007) | 1.655, 1.583, 1.528, 1.552 | 1.625 |
| corrected < -8 vs within | 0.771 | 0.36 | -0.005 (-0.044 to 0.033) | 0.817, 0.772, 0.702, 0.812 | 0.774 |
| matched older vs within | 1.538 | 0.11 | -0.020 (-0.049 to 0.008) | 1.613, 1.516, 1.512, 1.452 | 1.571 |
| matched younger vs within | 0.784 | 0.45 | -0.002 (-0.042 to 0.038) | 0.832, 0.762, 0.728, 0.829 | 0.789 |
| gap per year (raw = corrected) | 1.0293 | 0.56 | -0.0007 (-0.0022 to 0.0008) | 1.028, 1.031, 1.030, 1.026 | 1.0301 |
| age per year (continuous model) | 1.0755 | 0.28 | +0.0004 (-0.0007 to 0.0015) | 1.074, 1.075, 1.078, 1.076 | 1.0815 |
| male (continuous model) | 1.607 | 0.044 | -0.029 (-0.056 to -0.003) | 1.653, 1.644, 1.643, 1.387 | stratum |

Deaths per period: 2,812, 1,904, 2,443 and 1,182.

Older-looking against younger-looking directly (the contrast in the manuscript's ratio
table), by period and stratified:

| model | whole follow-up | 0-1 | 1-2 | 2-4 | 4+ | stratified |
|---|---|---|---|---|---|---|
| raw | 2.244 | 2.135 (1.898 to 2.402) | 2.406 (2.083 to 2.778) | 2.315 (2.031 to 2.638) | 2.128 (1.780 to 2.545) | 2.322 |
| corrected | 2.056 | 2.026 (1.802 to 2.278) | 2.051 (1.775 to 2.370) | 2.177 (1.910 to 2.482) | 1.911 (1.597 to 2.286) | 2.099 |
| share-matched | 1.961 | 1.940 (1.724 to 2.182) | 1.988 (1.718 to 2.301) | 2.079 (1.824 to 2.369) | 1.752 (1.465 to 2.096) | 1.993 |

- **Sex is the one term with a detectable departure** (p 0.039 to 0.044 in the five
  manuscript models; the KM transform gives 0.026 to 0.029). Its hazard ratio is about 1.65
  for the first 4 years and 1.39 after 4 years. In the held-out models, fitted on 80% of
  patients, the same drift is not significant (p 0.26 and 0.33).
- **Age shows none** (p 0.18 to 0.45; HR 1.067 to 1.078 in every period).
- **Held-out models**: age and sex, p 0.26 and 0.26; age, sex and gap, p 0.45, 0.33 and 0.36
  (gap HR by period 1.029, 1.031, 1.029, 1.026).

### Does any of this change a conclusion?

No. TESTED against each conclusion the Cox models support:

- "The published adjusted hazard ratio replicates (1.774 against 1.79)": the older-looking HR
  is 1.65 to 1.89 in every period and 1.818 stratified, so it is not an average of opposing
  effects at different follow-up times.
- "Adjusted for age, even the raw gap ranks the two groups the same way round (2.244)": the
  older-against-younger contrast is above 2.1 in every period and 2.322 stratified.
- "The correction attenuates it rather than removing it": the corrected older-looking HR
  (1.53 to 1.66 by period) sits below the raw one in all four periods and above 1 in all
  four.
- "HR per year of gap 1.029, raw and corrected identical": 1.026 to 1.031 by period, 1.0301
  stratified.
- The held-out gain from the gap and the decision curve above rest on models with no
  detectable departure.

The one departure (sex) concerns an adjustment term. Stratifying on sex and age band, which
removes any assumption about how the sex effect behaves over time, moves the gap hazard
ratios by 0.1% to 3.5% (largest: raw older against younger, 2.244 to 2.322), none of them
across 1 and none enough to alter a comparison above.

## References

- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction
  models. Med Decis Making 2006;26:565-74. doi:10.1177/0272989X06295361
- Vickers AJ, Cronin AM, Elkin EB, Gonen M. Extensions to decision curve analysis, a novel
  method for evaluating diagnostic tests, prediction models and molecular markers. BMC Med
  Inform Decis Mak 2008;8:53. doi:10.1186/1472-6947-8-53
- McLernon DJ, Giardiello D, Van Calster B, Wynants L, van Geloven N, van Smeden M,
  Therneau T, Steyerberg EW. Assessing performance and clinical usefulness in prediction
  models with survival outcomes: practical guidance for Cox proportional hazards models.
  Ann Intern Med 2023;176(1):105-14. PMID 36571841. doi:10.7326/M22-0844
- Van Calster B, et al. Lancet Digit Health 2025;7(12):100916. PMID 41391983 (as cited in
  the brief; not re-read here).
- Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted
  residuals. Biometrika 1994;81(3):515-26. doi:10.1093/biomet/81.3.515
- Schoenfeld D. Partial residuals for the proportional hazards regression model.
  Biometrika 1982;69(1):239-41. doi:10.1093/biomet/69.1.239

The DOIs and the McLernon author list are from memory and were not re-fetched on 12 Nov
2026 (the PMIDs for McLernon and Van Calster are from the brief); check them before
submission.
