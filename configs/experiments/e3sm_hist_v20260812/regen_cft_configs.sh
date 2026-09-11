#!/bin/bash
# Regenerate the five committed stage-3 (-CFT) run configs from their parents'
# own config.yaml and best_inference_ckpt.tar, through make_cpl_config.py, so
# any change to the generator (2026-09-09: the coupling-interface fixes)
# reaches every run. Parents are read from the existing run yaml.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$D/../../../.venv/bin/python}
for f in "$D"/runs/E*-CFT*.yaml; do
  ocn=$(grep -o "/pscratch[^ ]*aug26-ft/E1[0-9]-FT[^ ]*best_inference_ckpt.tar" "$f" | head -1)
  atm=$(grep -o "/pscratch[^ ]*aug26-ft/E0[0-9]-FT[^ ]*best_inference_ckpt.tar" "$f" | head -1)
  [ -n "$ocn" ] && [ -n "$atm" ] || { echo "cannot find parents in $f"; exit 1; }
  ocn_cfg=$(dirname "$(dirname "$ocn")")/config.yaml
  atm_cfg=$(dirname "$(dirname "$atm")")/config.yaml
  "$PY" "$D/make_cpl_config.py" --atm-config "$atm_cfg" --ocn-config "$ocn_cfg" \
      --atm-ckpt "$atm" --ocn-ckpt "$ocn" --nodes 8 --out "$f" "$@"
done
