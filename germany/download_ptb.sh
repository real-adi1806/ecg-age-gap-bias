#!/bin/zsh
# PTB-XL 1.0.3 (CC BY 4.0), 500 Hz records only, from PhysioNet's open S3 mirror; every file checked against SHA256SUMS.txt.
PTBXL_DIR=${PTBXL_DIR:-$HOME/datasets/ptb-xl}
mkdir -p "$PTBXL_DIR" && cd "$PTBXL_DIR"
B=https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3
for f in SHA256SUMS.txt ptbxl_database.csv scp_statements.csv LICENSE.txt; do [[ -f $f ]] || curl -sS -L -O $B/$f; done
grep " records500/" SHA256SUMS.txt | awk '{print $2}' > r500.txt
[[ -f ptb.done ]] || { xargs -P 48 -I{} sh -c '[ -f {} ] || curl -sS -L --retry 5 --create-dirs -o {} '$B'/{}' < r500.txt; }
grep " records500/" SHA256SUMS.txt | sed 's/ /  /' | shasum -a 256 -c 2>&1 | grep -v ": OK" && { echo "CHECKSUM FAILURES"; exit 1; }
touch ptb.done; echo PTB_OK
