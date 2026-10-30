# Germany: does the Brazil correction transfer to PTB-XL?

The parent folder replicates the age-bias correction of the AI ECG age gap on the Brazilian CODE-15% cohort. This folder asks the question a site adopting that correction would have to answer: **if the correction line is fitted in one country, does it still remove the bias in another?**

The test cohort is the German PTB-XL dataset. The published age network of Lima et al. (2021) is run unchanged on the raw PTB-XL waveforms, and the line fitted in Brazil (`../results/results.json`) is applied without refitting. PTB-XL has no outcomes, so everything here is about the bias of the gap, never about survival.

**Not a medical device. Not for clinical use.**

## What it reports

For the first ECG of each German patient, under the Brazil line: the correlation of the corrected gap with age, its level, its shape across age bands, the difference between men and women, and the share of patients past the plus or minus 8 year thresholds, each with a 95% interval from 2,000 bootstrap resamples of patients. Then:

- a German line fitted on a random half of German patients and tested on the other half, alone and with Barthels' age-level step;
- every ECG instead of the first per patient;
- an input amplitude scale matched to CODE-15%, because the network's README gives the input unit two ways;
- the Brazil line fitted on a random half of the Brazilian cohort;
- both cohorts cut to ages 18 to 85, the range both cover;
- each sex corrected by the line fitted within that sex in Brazil;
- exploratory: ECGs with and without PTB-XL's NORM statement.

Every number is in `results/germany.json`. The console summary of the recorded run is `expected-output/run-2026-10-30.txt`.

## Check the numbers without the data

```
python verify_germany.py
```

Standard library only. It asserts every Germany number quoted in the accompanying manuscript against the committed `results/` and fails on any difference. CI runs it on every push. With `PAPER_A_MANUSCRIPT=path/to/manuscript.md` set, it also fails if a number is not printed in the manuscript.

The results files are committed here, unlike the parent folder's, because regenerating them needs about 49 GB of downloads.

## Run it

```
./run.sh
```

macOS with zsh and [uv](https://docs.astral.sh/uv/). It uses the Apple GPU through torch when present, otherwise the CPU. One command:

1. makes a Python 3.12 environment with pinned packages if there is none;
2. clones the network code at the commit used (f192e49) and downloads the published weights (md5 checked) into `data/`;
3. runs `../ecg_age_bias.py` if the Brazil results file is missing;
4. downloads PTB-XL 1.0.3 (500 Hz records, every file checked against its SHA256SUMS.txt) and CODE-15% (18 parts plus exams.csv, md5 checked), resumably, skipping what is already there;
5. extracts the network's predictions and final-layer features (cached per part in `data/features/`);
6. runs `analyze.py`, `run_germany.py` (which re-runs the network on PTB-XL from the raw waveforms rather than trusting the cache) and `verify_germany.py`.

From scratch it downloads about 49 GB and keeps about 64 GB under `~/datasets/` (override with `CODE15_DIR` and `PTBXL_DIR`). With the data on disk it makes no network calls and took under five minutes on an Apple M5 laptop. Nothing in `data/` or `.venv/` is committed.

## Files

| File | What it does |
|---|---|
| `run.sh` | the one command |
| `download_code15.sh`, `download_ptb.sh` | resumable, checksum-checked downloads |
| `extract_code.py`, `extract_loop.sh`, `extract_ptb.py` | the network's predictions and features, first exam per CODE-15% patient and every analysed PTB-XL ECG |
| `common.py` | data locations, PTB-XL loading (500 to 400 Hz, padded to 4,096 samples, millivolts), the network |
| `run_germany.py` | the Germany analysis; writes `results/germany.json` |
| `verify_germany.py` | asserts every quoted number against `results/` |
| `analyze.py` | an earlier exploratory step: linear heads refitted on the frozen network's features, a different model from the network's own output, quoted in the manuscript only as a labelled note; writes `results/results.json` and `results/results.md` |

## Input scale

PTB-XL is stored in millivolts and is fed to the network unchanged, following the network README's instruction to multiply volts by 1,000. The same README sentence also gives the scale as 1e-4 V. `run_germany.py` therefore measures the amplitude of every lead in both datasets and re-runs the network with PTB-XL scaled to match CODE-15%. Some results depend on this choice; `results/germany.json` holds both.

## Data and licences

Nothing below is redistributed here; `run.sh` fetches each from its source.

- Network code: [antonior92/ecg-age-prediction](https://github.com/antonior92/ecg-age-prediction), commit f192e49, MIT licence (copyright 2021 Emilly M Lima and Antônio H Ribeiro). Its notice travels with the clone in `data/third_party/`.
- Network weights: Zenodo [10.5281/zenodo.4892365](https://doi.org/10.5281/zenodo.4892365), CC BY 4.0.
- CODE-15%: Zenodo [10.5281/zenodo.4916206](https://doi.org/10.5281/zenodo.4916206), CC BY 4.0. Its patients were held out from the network's training.
- PTB-XL 1.0.3: PhysioNet [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45), CC BY 4.0.

The code in this folder is MIT, as is the rest of the repository.
