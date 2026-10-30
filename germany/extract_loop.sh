#!/bin/zsh
# Extract each CODE part as soon as its hdf5 is ready (the .done marker written by download_code15.sh).
cd "$(dirname "$0")"; source .venv/bin/activate
CODE15_DIR=${CODE15_DIR:-$HOME/datasets/code15}
for k in {0..17}; do
  while [[ ! -f $CODE15_DIR/exams_part$k.zip.done ]]; do sleep 20; done
  python extract_code.py $k
done
echo EXTRACT_ALLDONE
