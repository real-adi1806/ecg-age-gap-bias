# ecg-age-gap-bias

[![verify](https://github.com/real-adi1806/ecg-age-gap-bias/actions/workflows/verify.yml/badge.svg)](https://github.com/real-adi1806/ecg-age-gap-bias/actions/workflows/verify.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python 3.9+, no dependencies](https://img.shields.io/badge/python-3.9%2B%2C%20no%20dependencies-blue.svg)](requirements.txt)

**Age-regression bias in the AI ECG age gap: a replication on CODE-15%.**

**The AI ECG age gap is confounded by chronological age itself (r = -0.3842 on 233,647 Brazilian patients), and the confound is large enough to reverse the crude survival result.** Split this cohort the way the 2021 paper that made the biomarker famous split it, and the older-looking hearts die *less* (3.02% against 4.57%), because the raw split is really sorting people by age: mean age 42.2 in the older-looking group against 62.9 in the younger-looking one. Regress the gap on age, keep the residual, and the direction reverses to 5.14% against 2.57% with the age imbalance gone (52.7 against 52.2).

**This is a replication, not a discovery.** Correcting the AI-ECG predicted age for its regression on chronological age was applied by Hirota et al. on a Japanese cohort ([doi 10.1016/j.ijcha.2023.101172](https://doi.org/10.1016/j.ijcha.2023.101172), 2023) and, as a residual of the gap on age with mortality as the outcome, by Barthels et al. on a Belgian cohort ([doi 10.1093/ehjdh/ztaf137](https://doi.org/10.1093/ehjdh/ztaf137), first published 28 November 2025), who reported r = -0.54 and the same reversal. What is repeated here is Barthels' method on CODE-15%. What is not in Barthels et al. or the 2021 paper is section 4 below.

![Crude death rate for the older-looking and younger-looking groups, before and after correcting the age gap for chronological age](figure.svg)

**Not a medical device. Not for clinical use.** This is a secondary analysis of a public research dataset. It says nothing about any individual's health or risk.

---

## Quick start

```
git clone https://github.com/real-adi1806/ecg-age-gap-bias
cd ecg-age-gap-bias
python ecg_age_bias.py
```

That is the whole thing. It downloads the 35.5 MB dataset from Zenodo if it is
not already present, verifies its md5 against the published checksum, refuses to
continue if the bytes are wrong, and prints every number in this README. It
takes about 10 seconds after the download and needs **no third-party packages at
all**: no numpy, no pandas, no GPU. Every file parses under the Python 3.9
grammar and the whole suite was run end to end on **3.11.15 and 3.13.7**.

```
python cox_model.py      # age- and sex-adjusted Cox models, about 40 seconds
python verify.py         # assert every number below, then check the repository
```

`verify.py` is the reason to trust this page. The headline figures quoted here
(49 of them) are listed inside it against the code path that produces it, so a
README that drifts away from the code fails the check instead of quietly lying.
The rest are printed by the same code. It also fails if the Cox implementation's
own self-tests fail.

---

## 1. The confound

| | |
|---|---|
| Patients analysed | 233,647 (one row per patient, the first exam) |
| Deaths | 8,341 (3.57%) |
| Median follow-up | 3.46 years |
| Mean absolute error of the age model | 8.45 years |
| **r(gap, chronological age)** | **-0.3842** |
| Fitted | gap = -0.2131 x age + 11.9755 |
| r(corrected gap, chronological age) | -0.0000, zero by construction |

The gap, conventionally called predicted age deviation or PAD, is
`ECG-predicted age minus chronological age`. A biomarker of biological ageing
that is meant to be independent of how old you are turns out to carry a large
amount of exactly that information. This is the regression bias of any age model
fitted to minimise its error: with an 8 year error, its predictions are pulled
towards the mean age, so it over-predicts for the young and under-predicts for
the old.

## 2. What it does to the crude survival split

Thresholds are the 2021 paper's own, plus or minus 8 years, chosen there because
the model's mean absolute error is about 8 years.

| Group | n | deaths | death rate | 95% CI | mean age |
|---|---|---|---|---|---|
| **Raw gap** | | | | | |
| older-looking (> +8) | 56,400 | 1,702 | 3.02% | 2.88 to 3.16 | 42.2 |
| within 8 | 134,126 | 4,667 | 3.48% | 3.38 to 3.58 | 50.4 |
| younger-looking (< -8) | 43,121 | 1,972 | 4.57% | 4.38 to 4.77 | 62.9 |
| **Bias-corrected gap** | | | | | |
| older-looking (> +8) | 47,263 | 2,431 | 5.14% | 4.95 to 5.35 | 52.7 |
| within 8 | 139,150 | 4,698 | 3.38% | 3.28 to 3.47 | 49.6 |
| younger-looking (< -8) | 47,234 | 1,212 | 2.57% | 2.43 to 2.71 | 52.2 |

Older versus younger death rate ratio: **0.66 raw, 2.00 corrected.**

The mean age column is the whole argument. Before correction the two extreme
groups differ by 20.7 years of chronological age, so the comparison is not about
hearts at all. After correction they differ by 0.5 years.

## 3. The same thing with the published model, adjusted for age and sex

The rates above are crude, so they cannot be compared with the 2021 paper's
headline, which came from a Cox model adjusted for age and sex. `cox_model.py`
fits that model, in the standard library, so the comparison is like for like.

| Model | older-looking | younger-looking |
|---|---|---|
| 2021 paper, as published | HR 1.79 (1.69 to 1.90) | HR 0.78 (0.74 to 0.83) |
| **this repository, raw gap** | **HR 1.774 (1.674 to 1.880)** | **HR 0.791 (0.750 to 0.834)** |
| this repository, corrected gap | HR 1.586 (1.510 to 1.666) | HR 0.771 (0.724 to 0.822) |

The published estimates fall inside the intervals reproduced here and vice
versa, which is the point: **the published headline replicates, and the bias
correction then attenuates it rather than destroying it.**

Attenuation is quoted here on three scales, because "attenuated by X%" means
three different numbers and papers rarely say which one they mean. Raw against
corrected: the hazard ratio falls from 1.774 to 1.586, which is **10.6% on the
hazard ratio itself, 19.6% on the log hazard ratio, and 24.3% on the excess
hazard** (an excess of 77.4% becomes 58.6%).

**And one result worth more than any of the above.** Fit the gap as a continuous
covariate instead of thresholding it:

| Model | hazard ratio per year of gap |
|---|---|
| raw gap, adjusted for age and sex | 1.029307 |
| corrected gap, adjusted for age and sex | 1.029307 |

Identical to six decimal places, and they have to be. The corrected gap is the
raw gap minus a linear function of age, so in a model that already adjusts for
age linearly, the two are the same fit with the age coefficient moved. Barthels
et al. found the same: HR 1.020 per year for both in their age-adjusted model,
and about 1.4% for both after adding risk factors.

So the confound matters enormously for how this biomarker is **reported and
thresholded**, and not at all for whether a continuous age-adjusted model
detects it. Thresholded groups are a common way to report this biomarker (the
2026 meta-analysis notes that studies "adopted different definitions"), and this
distinction matters most there.

## 4. The part that is not in either paper

**The pooled correction is not sex fair.** Fit it on everyone, then look inside
each sex. If the correction were sex fair the mean corrected gap would be zero
in both.

| | mean corrected gap | n |
|---|---|---|
| male | **+0.372 years** | 94,736 |
| female | **-0.254 years** | 138,911 |
| residual gap | **0.626 years** | 233,647 |

Male ECGs read older. A correction fitted on the pooled cohort leaves that in
place, so every downstream threshold is applied to a quantity that is still
shifted by sex.

The slopes are nearly identical (-0.2100 in men, -0.2159 in women), so this is an
**intercept** difference, not a slope difference. The obvious fix is to correct
within sex, and this is what happens when you do:

| corrected within sex | older-looking | younger-looking | rate ratio |
|---|---|---|---|
| male | 6.57% | 3.48% | **1.89** |
| female | 4.03% | 2.05% | **1.96** |

**That is a null and it is reported here as one.** The mortality gradient is
about the same in both sexes, so correcting within sex fixes the calibration of
the biomarker without changing what it predicts. A negative result about a
fairness intervention is still a result, and it is cheaper to publish it than to
let three other groups each discover it privately.

## 5. Apparently normal ECGs

The 2021 abstract makes a specific claim that the gap predicts mortality even
among ECGs read as normal. That subgroup is in the data (n = 92,138, 1,358
deaths) and the confound is **worse** there, not better: r = -0.4160.

| | older-looking | younger-looking | rate ratio |
|---|---|---|---|
| raw gap | 1.15% | 1.94% | 0.59 |
| corrected within this subgroup | 2.27% | 1.11% | 2.04 |

The reversal holds in the subgroup the original abstract highlights.

---

## Is this actually the cohort the paper analysed?

Worth checking rather than assuming, because the answer is yes with a caveat
that matters.

| | 2021 paper, CODE-15% | this extraction |
|---|---|---|
| r(predicted age, chronological age) | 0.84 | **0.8381** |
| mean chronological age | 51 (sd 20) | 50.74 (sd 19.82) |
| mean predicted age | 52 (sd 19) | 51.90 (sd 18.61) |
| male | 41% | 40.5% |
| mean absolute error | 8.38 (sd 7.00) | 8.45 |
| **n** | **218,169** | **233,647** |

Four summary statistics match at the paper's rounding, and the mean absolute
error is close (8.45 against 8.38). **The sample size does not match**, and this README
will not paper over it: the published analysis used 218,169 participants and
this one uses all 233,647 patients in the public deposit who carry an outcome.
The paper does not state the exclusions that take it from one number to the
other. That is the most likely reason the reproduced hazard ratio is 1.774 and
not 1.790, and it is a limitation of this replication rather than a finding
about the original.

One trap for anyone re-deriving these numbers. `exams.csv` holds 345,779 exam
rows but only 233,647 patients with an outcome, because mortality and follow-up
are recorded only on each patient's first exam. Summary statistics over all rows
give mean age 53.24 and MAE 8.53; over the analysable first-exam rows they give
50.74 and 8.45. It is the second set that matches the published table, and the
second set that this analysis uses.

---

## Novelty, stated honestly

This is the section that decides whether the rest of the page is trustworthy, so
it names the instrument and the gaps rather than asserting a first.

**What was searched**, all on the Europe PMC REST API on 22 September 2026, and
all retrieved rather than recalled:

- The record and full text of the 2021 paper (PMC8387361), the Belgian
  correction paper (PMC12853119), a second bias paper in the same journal
  (PMC12994139), and the 2026 meta-analysis (PMC13398890). The Japanese paper
  (PMC9841236) was read afterwards, once the Belgian paper's discussion had
  pointed to it. Every word frequency
  quoted here was counted in that retrieved full text.
- The complete citation list of the 2021 paper: **172 citing records**, all
  retrieved. Their titles were pattern matched for bias, confounding,
  calibration, residual, correction, fairness, subgroup and sex vocabulary.
  **Five matched. Zero mention CODE-15.** Of the five, one is the correction
  paper being replicated here; three are about predicting *sex* from an ECG,
  which is a different task; and one is a 2024 narrative review of data bias in
  cardiology AI in general, which does not address the age-gap regression bias
  or the sex fairness of a correction for it.

**A caveat on that instrument, worth stating because it cost time.** Europe PMC
returns a bare version stamp, `{"version":"6.9"}`, with no hitCount and no error,
when a query trips it. **That is not a zero result**, and reading it as one is
how a literature search silently returns nothing. A query form that failed was
re-run successfully with `SRC:MED AND EXT_ID:` prefixing. The citations endpoint
returned HTTP 500 for `pageSize=1000` on 22 September and worked at 200; a later
check returned 200 at 1000 as well, so that failure may not recur.

**What was NOT reached, stated as plainly as what was.** Scopus, Web of Science,
Google Scholar and IEEE Xplore were not searched, and neither were preprint
servers. Across the 172 citing records only TITLES were matched, not abstracts
and not full texts, so a paper doing this correction without saying so in its
title would not have been found. That is what happened: this title screen missed
Hirota et al. 2023, which corrects for the bias without saying so in its title.
It was found through the discussion section of Barthels et al.

Europe PMC's citation graph is not the same as
Scopus's or Web of Science's and will be incomplete.

**Therefore.** The claim made here is the narrow one that can be supported:
*the published corrections were applied to Japanese and Belgian cohorts, not to
CODE-15%, and nothing found suggests a CODE-15% replication exists.*
That is a hypothesis consistent with what was searched. It is not an established
fact, and anyone submitting this should check Scopus first.

**One piece of timing that is checkable in a minute and is the most interesting
thing here.** The 2026 meta-analysis of this biomarker
([doi 10.36660/abc.20250650](https://doi.org/10.36660/abc.20250650)) pooled seven
studies (eleven analyses) of the *uncorrected* gap for all-cause mortality, from
a review of ten studies (over 550,000 participants, 336,026 after removing
overlapping datasets), and reported a pooled hazard ratio of 1.83 with
I-squared of 82.9%, which is substantial unexplained heterogeneity. It searched nine databases
including Scopus and Web of Science from inception to **1 May 2025**. The
correction paper was first published on **28 November 2025**, nearly seven months after
that search closed. In the meta-analysis full text, "regression bias", "confound",
"bias-correct" and "residual" each appear **zero** times.

Whether the confound explains any of that heterogeneity is an open question and
this repository does not answer it. Seven studies is barely enough for one
covariate in a meta-regression, and it may well return nothing.

**A correction to a tempting framing.** This is not a novel statistical idea in
any field. Regressing an age gap on chronological age is long standing practice
in the neuroimaging "brain age" literature, and any predicted-age-minus-real-age
biomarker in any organ carries the same bias. The contribution here is a cohort
and a subgroup result, not a method.

---

## What this does and does not show

**It does not show the 2021 paper is wrong.** That paper adjusted for age and
sex and its headline replicates here (section 3). The crude reversal in section
2 is a statement about the raw gap as a standalone quantity, which is how it is
frequently used downstream, and not about the published model.

**It does not show the biomarker is useless.** Section 3 shows the opposite: a
continuous age-adjusted model finds the same signal either way.

**It does show** that the raw gap is heavily age-confounded, that this reverses
the direction of any crude or thresholded analysis, that the age adjustment in
the published model is therefore carrying a great deal of weight, and that the
standard pooled correction leaves a 0.63 year sex offset behind.

**Design limits.** This is a secondary analysis of a single observational
cohort. No new model was trained: the age predictions are the original authors'
own, already a column in the published file, so nothing here says anything about
any other ECG age model. Crude rates ignore censoring; the Cox models in section
3 do not, and should be preferred. Deaths are from record linkage and cause of
death is not available. There is no external validation set in this repository.

## Using the correction on another cohort

The scripts are locked to the published `exams.csv` on purpose: they check its
md5 and refuse any other file, so the numbers on this page cannot come from a
different file by accident. The correction itself needs nothing from here. With
chronological age and predicted age for each person in your own cohort:

1. Compute the gap: predicted age minus chronological age.
2. Fit it on chronological age by ordinary least squares: `gap = a + b x age`.
3. Use the residual, `gap - (a + b x age)`, in place of the raw gap.

On CODE-15% the fit is a = 11.9755 and b = -0.2131 (section 1). Fit a and b on
your own cohort rather than reusing these. Then check what section 4 checks: a
fit on everyone can leave the corrected gap offset between men and women.

---

## Data

CODE-15%, from the Telehealth Network of Minas Gerais, Brazil.

> Ribeiro, A. H., Paixao, G. M. M., Lima, E. M., Horta Ribeiro, M. et al. *CODE-15%: a large
> scale annotated dataset of 12-lead ECGs.* Zenodo.
> https://doi.org/10.5281/zenodo.4916206 Licensed CC BY 4.0.

This repository contains no patient data and its MIT licence does not cover the
data. CC BY 4.0 permits reuse, including commercial reuse, with attribution; the
citation above is that attribution. Anyone running this code downloads the data
from Zenodo directly and is bound by the data's licence, not by this one.

Only `exams.csv` is used: 35,500,900 bytes, md5
`0107516d3f63864498fb77d15799cc95`. The 46 GB of raw waveforms in the same
deposit is not needed, because the neural network's own age predictions are
already a column.

**No data file is committed to this repository, and the `.gitignore` is written
so that it cannot be by accident.** A pre-commit hook that refuses data files
and anything over 1 MB ships in `.githooks/`; switch it on once per clone with
`git config core.hooksPath .githooks`.

## Files

| | |
|---|---|
| `ecg_age_bias.py` | the replication. Standard library only. Downloads, checksums, analyses, writes `results/results.json` |
| `cox_model.py` | Cox proportional hazards with Efron ties, also standard library. `--selftest` runs the correctness checks alone |
| `verify.py` | asserts every number in this README, then scans the repository for secrets, absolute paths, data files and em dashes |
| `make_figure.py` | draws `figure.svg` at the top of this page from `results/results.json`. Standard library only, reads numbers and computes nothing new |
| `crosscheck_lifelines.py` | optional. Re-fits the same models with lifelines and compares. The only reason `requirements.txt` is not empty |
| `Makefile` | optional convenience. Every target is one python command |

## Is the Cox implementation right?

It is written out rather than imported, so the question is fair. It is checked
four ways, and all four are reproducible from this repository.

1. **Against a naive implementation.** The fast cumulative-sum risk-set
   accumulation is compared with a transparent O(n squared) loop over the whole
   risk set at every event time. Worst relative disagreement across
   log-likelihood, gradient and information matrix: **3.3e-16**, which is
   machine precision. `python cox_model.py --selftest`
2. **Against the optimality conditions.** At the solution, max absolute score is
   **4.1e-09** and the information matrix is positive definite.
3. **Against lifelines.** Standard errors agree to **6.0e-09**. Standard errors
   are the sharp test because they come from the information matrix, a second
   derivative, which is unforgiving of any error in the likelihood. The fitted
   log-likelihood here is never lower than lifelines' own.
4. **Against a published number this code did not produce.** HR 1.774 against
   the paper's 1.79, on nearly the same cohort.

`python verify.py` runs 49 numeric assertions plus the hygiene scan and exits
non-zero if any of them moves. It was run to completion on Python 3.11.15 and
3.13.7, and on a fresh clone that downloaded the data itself.

On point 3, one detail is worth recording because the obvious version of that
check failed first. Comparing *coefficients* at 1e-6 fails, and the reason is
not arithmetic: evaluating both solutions in the same likelihood gives max score
4.1e-09 here against 2.1e-01 for lifelines, and a log-likelihood of
-94661.982283812 against -94661.982283830. lifelines stops slightly earlier, so
a coefficient tolerance measures its stopping rule rather than this code's
correctness. Both answers are right to every decimal anyone reports.

---

## Declarations

**Author.** Adithya Krishnan.

**Affiliation.** None at the time of this analysis. The author began an MBBS at
Barts and The London School of Medicine and Dentistry, Queen Mary University of
London, in September 2026. No institution has reviewed, sponsored or endorsed
this work.

**Funding.** None.

**Competing interests.** None.

**Ethics.** No new approval was sought and none is required. This is a secondary
analysis of a published, de-identified, openly licensed dataset containing no
identifiers. The original CODE study was approved by the Research Ethics
Committee of the Universidade Federal de Minas Gerais under protocol
49368496317.7.0000.5149, and that committee did not require informed consent for
secondary analysis of the anonymised database. Nothing here re-identifies anyone
or adds a participant.

**Contributions.** All analysis, code and text by the author. Large language
model assistance was used during drafting and coding. Every number computed
from CODE-15% is produced by the code in it and can be regenerated with one
command; numbers quoted from other papers were copied from their full texts and
are cited.

**Data availability.** The dataset is public at the DOI above and is not
redistributed here. Every result regenerates from it with `python verify.py`.

---

## References

1. Lima, E. M., Ribeiro, A. H., Paixao, G. M. M. et al. Deep neural
   network-estimated electrocardiographic age as a mortality predictor.
   *Nature Communications* **12**, 5117 (2021).
   [doi 10.1038/s41467-021-25351-7](https://doi.org/10.1038/s41467-021-25351-7).
   PMID 34433816, PMC8387361. Open access.
2. Barthels, M., Verhofstadt, E., Delgado, I. B. et al. Artificial
   intelligence-predicted ECG age gap as a biomarker: bias-adjusted correlation
   with mortality and cardiovascular risk factors. *European Heart Journal
   Digital Health* **7**(2) (2026), first published 28 November 2025.
   [doi 10.1093/ehjdh/ztaf137](https://doi.org/10.1093/ehjdh/ztaf137).
   PMID 41624565, PMC12853119. Open access. **The method repeated here.**
3. Vander Heyde, L., Dujardin, K., Anné, W. et al. Artificial
   intelligence-derived electrocardiographic age gap as a predictor of mortality
   after coronary revascularization: prognostic value and short-term
   intra-patient variability. *European Heart Journal Digital Health* **7**(3)
   (2026).
   [doi 10.1093/ehjdh/ztag039](https://doi.org/10.1093/ehjdh/ztag039).
   PMID 41853636, PMC12994139. Open access.
4. Bozzi, I. C. R. S., Lima, M. C. A. G., Ribeiro, A. L. P., Paixão, G. M. M.
   Artificial Intelligence-Derived ECG-Age as a Predictor of Mortality and
   Cardiovascular Events: A Systematic Review and Meta-Analysis. *Arquivos
   Brasileiros de Cardiologia* **123**(4) (2026).
   [doi 10.36660/abc.20250650](https://doi.org/10.36660/abc.20250650).
   PMID 42307380, PMC13398890. Open access. PROSPERO CRD420251042467.
5. Hirota, N., Suzuki, S., Motogi, J. et al. Cardiovascular events and
   artificial intelligence-predicted age using 12-lead electrocardiograms.
   *International Journal of Cardiology. Heart & Vasculature* **44**, 101172
   (2023). [doi 10.1016/j.ijcha.2023.101172](https://doi.org/10.1016/j.ijcha.2023.101172).
   PMID 36654885, PMC9841236. Open access. Applied a linear bias correction to
   the predicted age on a Japanese single-centre cohort.

## Citing it

GitHub's "Cite this repository" button (from `CITATION.cff`) gives the citation.
Please cite the dataset and the papers this replicates as well; they are listed
in the same file and under References above.

## Contributing

Bug reports and questions are welcome as issues. If a number here does not
reproduce on your machine, an issue with the output of `python verify.py` and
your Python version is the most useful thing to send.

## Licence

Code: MIT, see `LICENSE`. Data: CC BY 4.0, held by its original authors and not
redistributed here.
