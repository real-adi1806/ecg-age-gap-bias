#!/bin/zsh
# Resumable download of CODE-15% parts with md5 check; skips finished parts.
CODE15_DIR=${CODE15_DIR:-$HOME/datasets/code15}
mkdir -p "$CODE15_DIR" && cd "$CODE15_DIR"
curl -s https://zenodo.org/api/records/4916206 | python3 -c "
import json,sys
d=json.load(sys.stdin)
for f in d['files']: print(f['key'],f['checksum'].split(':')[1],f['links']['self'])" > files.txt
sort -V files.txt | while read k md5 url; do
  [[ "$k" == *.zip ]] && base=${k%.zip}.hdf5 || base=$k
  if [[ -f "$k.done" ]]; then continue; fi
  curl -L -C - --retry 5 --retry-delay 5 -sS -o "$k" "$url"
  got=$(md5 -q "$k")
  if [[ "$got" != "$md5" ]]; then echo "MD5 MISMATCH $k $got $md5" >> download.log; continue; fi
  echo "OK $k $(stat -f%z "$k") $got" >> download.log
  if [[ "$k" == *.zip ]]; then unzip -oq "$k" -d . && rm "$k" && touch "$k.done"; else touch "$k.done"; fi
done
echo ALLDONE >> download.log
