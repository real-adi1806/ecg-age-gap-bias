#!/bin/zsh
# One command, end to end: environment, data, features, the exploratory heads (analyze.py), the Germany analysis
# (run_germany.py) and the assertions on every Germany number the manuscript quotes (verify_germany.py).
# With the data already on disk it makes no network calls: downloads run only for what is missing.
# From scratch: about 49 GB downloaded, see README.md. With the data on disk: see expected-output/ for the timing.
set -euo pipefail
cd "$(dirname "$0")"
export CODE15_DIR=${CODE15_DIR:-$HOME/datasets/code15}
export PTBXL_DIR=${PTBXL_DIR:-$HOME/datasets/ptb-xl}
NET_COMMIT=f192e49e095708c038e4f430a41ad31fc4d3f8bd   # antonior92/ecg-age-prediction, the commit this was run with
[[ -d .venv ]] || { uv venv --python 3.12 && uv pip install torch==2.14.1 h5py==3.16.0 numpy==2.5.4 pandas==3.1.0 \
  scipy==1.18.1 wfdb==4.3.1 lifelines==0.30.0; }
mkdir -p data/features data/model data/third_party
if [[ ! -d data/third_party/ecg-age-prediction ]]; then
  git clone https://github.com/antonior92/ecg-age-prediction data/third_party/ecg-age-prediction
  git -C data/third_party/ecg-age-prediction checkout -q $NET_COMMIT
fi
if [[ ! -f data/model/model/model.pth ]]; then
  curl -L -C - --retry 5 -o data/model/model.zip https://zenodo.org/api/records/4892365/files/model.zip/content
  [[ $(md5 -q data/model/model.zip) == 00aad2c0341942837019086cf5d94cd5 ]] || { echo "model.zip md5 mismatch"; exit 1; }
  (cd data/model && unzip -oq model.zip)
fi
source .venv/bin/activate
# The Brazil line is read from the replication's own results file in the parent folder.
[[ -f ../results/results.json ]] || (cd .. && python ecg_age_bias.py)
[[ -f $PTBXL_DIR/ptb.done ]] || ./download_ptb.sh
n_code=$(find "$CODE15_DIR" -maxdepth 1 -name 'exams_part*.zip.done' 2>/dev/null | wc -l | tr -d ' ')
if [[ $n_code -ne 18 || ! -f $CODE15_DIR/exams.csv.done ]]; then
  ./download_code15.sh &
  ./extract_loop.sh      # extracts each CODE part as it lands
  wait
fi
python extract_code.py   # cached per part: a re-run only checks the cache
python extract_ptb.py    # cached
python analyze.py        # 25 Oct exploratory analysis with retrained heads; a labelled supplement note in the manuscript
python run_germany.py    # the network re-run on PTB-XL, the Brazil line applied unchanged
python verify_germany.py
